/**
 * Recursive V2 bot detector — recursive strategy with v3's minimal prompts.
 *
 * Strategy (same as recursive.ts):
 *   1. First N passes: classify all users with a biased prompt (lean BOT).
 *      If the model says HUMAN, we're very confident. Each pass filters out more humans.
 *   2. Final pass: run a minimal baseline classification on only the remaining suspects.
 *
 * Can be run standalone:  bun run detectors/recursive-v2.ts [--model <model>] [--delay <ms>] <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/recursive-v2.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { MODEL_DEEPSEEK } from "../models.ts";

export const RECURSION_DEPTH = 8;
export const TOTAL_ROUNDS = RECURSION_DEPTH + 1;

// ── Biased prompt (rounds 0..RECURSION_DEPTH-1) — v3 style + bias ───

const BIASED_SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

You have a STRONG bias toward labeling accounts as BOT. Only label an account as HUMAN if you are very confident it is a real person. When in doubt, label it as BOT.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus
- Political
- Inciting anger or hate by being grossly ignorant.

Respond with ONLY "BOT" or "HUMAN" — nothing else.`;

// ── Baseline prompt (final round) — v3 style ─────────────────────────

const BASELINE_SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus
- Political
- Inciting anger or hate by being grossly ignorant.

+4 for a correctly flagged bot, -1 for a missed bot, -2 for a wrongly flagged human.

Respond with ONLY "BOT" or "HUMAN" — nothing else.`;

// ── Prompt builder (v3 style) ────────────────────────────────────────

function buildUserPrompt(user: User, posts: Post[]): string {
  const profile = [
    `Username: ${user.username}`,
    `Name: ${user.name}`,
    `Description: ${user.description || "(none)"}`,
    `Location: ${user.location || "(none)"}`,
    `Tweet count: ${user.tweet_count}`,
    `Z-score (posting activity deviation from average): ${user.z_score.toFixed(4)}`,
  ].join("\n");

  const postTexts = posts
    .slice(0, 50)
    .map((p) => `[${p.created_at}] ${p.text}`)
    .join("\n");

  return `${profile}\n\n${postTexts}`;
}

// ── Types ────────────────────────────────────────────────────────────

export type DetectorResult = {
  runFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

export type CompletedRound = {
  round: number;
  label: string;
  poolSize: number;
  botsOut: number;
  humansFiltered: number;
};

export type RecursiveProgress = {
  currentRound: number;
  totalRounds: number;
  roundDone: number;
  roundTotal: number;
  roundLabel: string;
  completedRounds: CompletedRound[];
};

// ── Main detector ────────────────────────────────────────────────────

export async function runDetector(
  datasetIds: number[],
  model = MODEL_DEEPSEEK,
  onProgress?: (done: number, total: number) => void,
  delayMs = 50,
  cacheFile?: string,
  onRecursiveProgress?: (progress: RecursiveProgress) => void,
): Promise<DetectorResult> {
  const paths = datasetIds.map((id) => `datasets/dataset.posts&users.${id}.json`);
  const datasets = await loadDatasets(paths);

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

  let cachePath: string;
  let cacheData: CacheData;

  if (cacheFile) {
    cacheData = await loadCache(cacheFile);
    cachePath = cacheFile;
  } else {
    cachePath = await createCache("recursive-v2", model, datasetIds, users.length * TOTAL_ROUNDS);
    cacheData = await loadCache(cachePath);
  }

  const cacheKey = (round: number, userId: string) =>
    round < RECURSION_DEPTH ? `round${round}:${userId}` : `final:${userId}`;

  const completedRounds: CompletedRound[] = [];
  let currentPool = users;

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

  emit(0, 0, users.length, "Biased filter");
  onProgress?.(0, users.length);

  // ── Biased rounds ──────────────────────────────────────────────────

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
          const myDelay = apiIndex++;
          if (delayMs > 0 && myDelay > 0) await Bun.sleep(delayMs * myDelay);
          const posts = getPostsByUserFromLoaded(u.id, datasets);
          const prompt = buildUserPrompt(u, posts);
          const answer = await chatWithSystem(BIASED_SYSTEM_PROMPT, prompt, model);
          const isBot = answer.trim().toUpperCase().startsWith("BOT");
          cacheData.results[key] = { isBot };
          await writeResult(cachePath, cacheData);
        }
        roundDone++;
        emit(round, roundDone, roundTotal, "Biased filter");
        const totalDone = completedRounds.reduce((s, r) => s + r.poolSize, 0) + roundDone;
        onProgress?.(totalDone, users.length);
      }),
    );

    const bots = currentPool.filter((u) => cacheData.results[cacheKey(round, u.id)]?.isBot);
    const humansFiltered = roundTotal - bots.length;
    completedRounds.push({ round, label: "Biased filter", poolSize: roundTotal, botsOut: bots.length, humansFiltered });
    currentPool = bots;
  }

  // ── Final baseline round ───────────────────────────────────────────

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
          const myDelay = apiIndex++;
          if (delayMs > 0 && myDelay > 0) await Bun.sleep(delayMs * myDelay);
          const posts = getPostsByUserFromLoaded(u.id, datasets);
          const prompt = buildUserPrompt(u, posts);
          const answer = await chatWithSystem(BASELINE_SYSTEM_PROMPT, prompt, model);
          const isBot = answer.trim().toUpperCase().startsWith("BOT");
          cacheData.results[key] = { isBot };
          await writeResult(cachePath, cacheData);
        }
        roundDone++;
        emit(finalRound, roundDone, roundTotal, "Final baseline");
        const totalDone = completedRounds.reduce((s, r) => s + r.poolSize, 0) + roundDone;
        onProgress?.(totalDone, users.length);
      }),
    );

    const bots = currentPool.filter((u) => cacheData.results[cacheKey(finalRound, u.id)]?.isBot);
    const humansFiltered = roundTotal - bots.length;
    completedRounds.push({ round: finalRound, label: "Final baseline", poolSize: roundTotal, botsOut: bots.length, humansFiltered });
    currentPool = bots;
  }

  // ── Write results ──────────────────────────────────────────────────

  const bots = currentPool;

  cacheData.totalUsers = Object.keys(cacheData.results).length;
  await writeResult(cachePath, cacheData);

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `runs/recursive-v2-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: recursive-v2`,
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

  let model = MODEL_DEEPSEEK;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_DEEPSEEK;
    args.splice(modelIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/recursive-v2.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Recursive V2 | Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms | Depth: ${RECURSION_DEPTH} + final`);

  const result = await runDetector(datasetIds, model, undefined, delayMs, undefined, (p) => {
    const roundLabel = `${p.roundLabel} [${p.currentRound + 1}/${p.totalRounds}]`;
    const pct = p.roundTotal > 0 ? Math.round((p.roundDone / p.roundTotal) * 100) : 0;
    process.stdout.write(`\r${roundLabel}: ${p.roundDone}/${p.roundTotal} (${pct}%)   `);
    if (p.roundDone === p.roundTotal && p.completedRounds.length > 0) {
      const last = p.completedRounds[p.completedRounds.length - 1]!;
      process.stdout.write(`→ ${last.botsOut} bots, ${last.humansFiltered} filtered`);
    }
  });

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
