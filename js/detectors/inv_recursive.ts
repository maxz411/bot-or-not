/**
 * Inverse recursive bot detector — biased against false positives (high recall for humans).
 *
 * Strategy:
 *   1. First N passes: classify all users with a prompt that errs on the side of
 *      labeling users as HUMAN. If the model says BOT, we're very confident.
 *      Each pass filters out more bots from the HUMAN pool.
 *   2. Final pass: run a normal baseline classification (baseline prompt, no bias)
 *      on only the remaining suspects for a fair final determination.
 *
 * Caching is transparent — classify functions check the cache first and only
 * call the API on a miss. Resuming a run just means running the same loop
 * again; cached results resolve instantly.
 *
 * Can be run standalone:  bun run detectors/inv_recursive.ts <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/inv_recursive.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { MODEL_OPENAI } from "../models.ts";
import { datasetPaths, RUNS_DIR } from "../paths.ts";

export const RECURSION_DEPTH = 8;
/** Total rounds = RECURSION_DEPTH biased + 1 final baseline */
export const TOTAL_ROUNDS = RECURSION_DEPTH + 1;

// ── Biased prompt (rounds 0..RECURSION_DEPTH-1) ─────────────────────

const BIASED_SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

You have a STRONG bias toward labeling accounts as HUMAN. Only label an account as BOT if you are very confident it is a bot. When in doubt, label it as HUMAN.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus
- Any slight doubt about bot-ness should tip the scale toward HUMAN

Respond with ONLY "BOT" or "HUMAN" — nothing else.`;

// ── Baseline prompt (final round) ─────────────────────────────────────

const BASELINE_SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus

Respond with ONLY "BOT" or "HUMAN" — nothing else.`;

// ── Prompt builders ──────────────────────────────────────────────────

function buildUserPrompt(user: User, posts: Post[]): string {
  const profile = [
    `Username: ${user.username}`,
    `Name: ${user.name}`,
    `Description: ${user.description || "(none)"}`,
    `Location: ${user.location || "(none)"}`,
    `Tweet count: ${user.tweet_count}`,
    `Z-score: ${user.z_score.toFixed(4)}`,
  ].join("\n");

  const postTexts = posts
    .slice(0, 50)
    .map((p) => `[${p.created_at}] ${p.text}`)
    .join("\n");

  return `USER PROFILE:\n${profile}\n\nPOSTS (${posts.length} total, showing up to 50):\n${postTexts}`;
}


// ── Types ────────────────────────────────────────────────────────────

export type DetectorResult = {
  runFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

/** Info about a completed recursion round. */
export type CompletedRound = {
  round: number;          // 0-indexed
  label: string;          // e.g. "Biased filter" or "Final batch"
  poolSize: number;       // users entering this round
  botsOut: number;        // labeled BOT (removed from pool / filtered out)
  humansFiltered: number; // labeled HUMAN (kept in pool)
};

/** Progress for the inverse recursive detector — emitted on every classification. */
export type InvRecursiveProgress = {
  currentRound: number;   // 0-indexed round being worked on
  totalRounds: number;    // TOTAL_ROUNDS
  roundDone: number;      // classifications done in current round
  roundTotal: number;     // pool size for current round
  roundLabel: string;     // label for the current round
  completedRounds: CompletedRound[];
};

// ── Main detector ────────────────────────────────────────────────────

export async function runDetector(
  datasetIds: number[],
  model = MODEL_OPENAI,
  onProgress?: (done: number, total: number) => void,
  delayMs = 50,
  cacheFile?: string,
  onRecursiveProgress?: (progress: InvRecursiveProgress) => void,
): Promise<DetectorResult> {
  const paths = datasetPaths(datasetIds);
  const datasets = await loadDatasets(paths);

  // Collect all unique users
  const seen = new Set<string>();
  const users: User[] = [];
  for (const ds of datasets) {
    for (const u of ds.users) {
      if (!seen.has(u.id)) {
        seen.add(u.id);
        users.push(u);
      }
    }
  }

  // Set up cache — totalUsers is set high so incomplete runs show in resume list
  let cachePath: string;
  let cacheData: CacheData;

  if (cacheFile) {
    cacheData = await loadCache(cacheFile);
    cachePath = cacheFile;
  } else {
    cachePath = await createCache("inv_recursive", model, datasetIds, users.length * TOTAL_ROUNDS);
    cacheData = await loadCache(cachePath);
  }

  const cacheKey = (round: number, userId: string) =>
    round < RECURSION_DEPTH ? `round${round}:${userId}` : `final:${userId}`;

  const completedRounds: CompletedRound[] = [];
  let currentPool = users;

  // Track bots filtered out across all biased rounds
  const filteredBotIds = new Set<string>();

  // Helper to emit recursive progress
  const emit = (round: number, roundDone: number, roundTotal: number, label: string) => {
    onRecursiveProgress?.({
      currentRound: round,
      totalRounds: TOTAL_ROUNDS,
      roundDone,
      roundTotal,
      roundLabel: label,
      completedRounds: [...completedRounds],
    });
  };

  // Emit initial progress so the UI shows totals immediately (important for resume)
  emit(0, 0, users.length, "Biased filter");
  onProgress?.(0, users.length);

  // ── Biased rounds — individual classification ──────────────────────

  for (let round = 0; round < RECURSION_DEPTH; round++) {
    if (currentPool.length === 0) break;

    const roundTotal = currentPool.length;
    let roundDone = 0;
    let apiIndex = 0;

    emit(round, 0, roundTotal, "Biased filter");

    await Promise.all(
      currentPool.map(async (u) => {
        const key = cacheKey(round, u.id);
        if (!cacheData.results[key]) {
          // Cache miss — call API with staggered delay
          const myDelay = apiIndex++;
          if (delayMs > 0 && myDelay > 0) await Bun.sleep(delayMs * myDelay);
          const posts = getPostsByUserFromLoaded(u.id, datasets);
          const prompt = buildUserPrompt(u, posts);
          const answer = await chatWithSystem(BIASED_SYSTEM_PROMPT, prompt, model);
          const isBot = answer.trim().toUpperCase().startsWith("BOT");
          cacheData.results[key] = { isBot };
          await writeResult(cachePath, cacheData);
        }
        // Cache hit or just-classified — count it
        roundDone++;
        emit(round, roundDone, roundTotal, "Biased filter");
        const totalDone = completedRounds.reduce((s, r) => s + r.poolSize, 0) + roundDone;
        onProgress?.(totalDone, users.length);
      }),
    );

    // Tally round results — keep humans in pool, filter out bots
    const humans = currentPool.filter((u) => !cacheData.results[cacheKey(round, u.id)]?.isBot);
    const botsFiltered = roundTotal - humans.length;
    // Track the bot IDs filtered out this round
    for (const u of currentPool) {
      if (cacheData.results[cacheKey(round, u.id)]?.isBot) {
        filteredBotIds.add(u.id);
      }
    }
    completedRounds.push({ round, label: "Biased filter", poolSize: roundTotal, botsOut: botsFiltered, humansFiltered: humans.length });
    currentPool = humans;
  }

  // ── Final baseline round — baseline prompt, no bias ────────────────

  if (currentPool.length > 0) {
    const finalRound = RECURSION_DEPTH;
    const roundTotal = currentPool.length;
    let roundDone = 0;
    let apiIndex = 0;

    emit(finalRound, 0, roundTotal, "Final baseline");

    await Promise.all(
      currentPool.map(async (u) => {
        const key = cacheKey(finalRound, u.id);
        if (!cacheData.results[key]) {
          // Cache miss — call API with staggered delay
          const myDelay = apiIndex++;
          if (delayMs > 0 && myDelay > 0) await Bun.sleep(delayMs * myDelay);
          const posts = getPostsByUserFromLoaded(u.id, datasets);
          const prompt = buildUserPrompt(u, posts);
          const answer = await chatWithSystem(BASELINE_SYSTEM_PROMPT, prompt, model);
          const isBot = answer.trim().toUpperCase().startsWith("BOT");
          cacheData.results[key] = { isBot };
          await writeResult(cachePath, cacheData);
        }
        // Cache hit or just-classified — count it
        roundDone++;
        emit(finalRound, roundDone, roundTotal, "Final baseline");
        const totalDone = completedRounds.reduce((s, r) => s + r.poolSize, 0) + roundDone;
        onProgress?.(totalDone, users.length);
      }),
    );

    // Tally round results
    const humans = currentPool.filter((u) => !cacheData.results[cacheKey(finalRound, u.id)]?.isBot);
    const botsFiltered = roundTotal - humans.length;
    completedRounds.push({ round: finalRound, label: "Final baseline", poolSize: roundTotal, botsOut: botsFiltered, humansFiltered: humans.length });
  }

  // ── Collect all bots ──────────────────────────────────────────────

  // Bots = those filtered during biased rounds + those labeled BOT in final round
  const finalRound = RECURSION_DEPTH;
  for (const u of currentPool) {
    if (cacheData.results[cacheKey(finalRound, u.id)]?.isBot) {
      filteredBotIds.add(u.id);
    }
  }

  const bots = users.filter((u) => filteredBotIds.has(u.id));

  // Mark cache complete (totalUsers = actual entry count → no longer shows as incomplete)
  cacheData.totalUsers = Object.keys(cacheData.results).length;
  await writeResult(cachePath, cacheData);

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `${RUNS_DIR}/inv_recursive-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: inv_recursive`,
    `Model: ${model}`,
    `Recursion depth: ${RECURSION_DEPTH}`,
    `Final round: baseline`,
    "",
    ...bots.map((r) => r.id),
  ];
  await Bun.write(runFile, lines.join("\n") + "\n");

  return {
    runFile,
    totalUsers: users.length,
    botsDetected: bots.length,
    humansDetected: users.length - bots.length,
  };
}

// ── CLI entry point ──────────────────────────────────────────────────

if (import.meta.main) {
  const args = Bun.argv.slice(2);

  let delayMs = 50;
  const delayIdx = args.indexOf("--delay");
  if (delayIdx !== -1) {
    delayMs = parseInt(args[delayIdx + 1] ?? "50", 10);
    args.splice(delayIdx, 2);
  }

  let model = MODEL_OPENAI;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_OPENAI;
    args.splice(modelIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/inv_recursive.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    console.log("Example: bun run detectors/inv_recursive.ts --model openai/gpt-5.1 --delay 100 30 31");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms | Depth: ${RECURSION_DEPTH} + final baseline`);

  const result = await runDetector(datasetIds, model, undefined, delayMs, undefined, (p) => {
    const roundLabel = `${p.roundLabel} [${p.currentRound + 1}/${p.totalRounds}]`;
    const pct = p.roundTotal > 0 ? Math.round((p.roundDone / p.roundTotal) * 100) : 0;
    process.stdout.write(`\r${roundLabel}: ${p.roundDone}/${p.roundTotal} (${pct}%)   `);
    if (p.roundDone === p.roundTotal && p.completedRounds.length > 0) {
      const last = p.completedRounds[p.completedRounds.length - 1]!;
      process.stdout.write(`→ ${last.humansFiltered} humans, ${last.botsOut} bots filtered`);
    }
  });

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
