/**
 * Stats-based bot detector — two-phase approach using empirical false positive rate.
 *
 * Phase 1: Classify all users with gemini-3-flash-preview (hardcoded) to get suspected bots.
 * Phase 2: Re-evaluate suspected bots with a configurable model, scoring confidence 0-100.
 *          Reclassify the bottom n (lowest confidence) as human, where n is derived from
 *          the empirical false discovery rate of phase 1.
 *
 * Can be run standalone:  bun run detectors/stats-based.ts [--model <model>] [--delay <ms>] <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/stats-based.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { MODEL_GEMINI } from "../models.ts";
import { datasetPaths, RUNS_DIR } from "../paths.ts";

// ── Constants ────────────────────────────────────────────────────────

/** Phase 1 always uses gemini-3-flash-preview so the FP rate stays calibrated. */
const PHASE1_MODEL = MODEL_GEMINI;

/**
 * Empirical false discovery rate: fraction of users flagged as bots that are
 * actually human. From v3 + gemini-3-flash-preview on dataset 30:
 *   22 FP out of 86 flagged (64 TP + 22 FP) = 25.58%
 */
const FALSE_DISCOVERY_RATE = 22 / 86;

// ── Prompts ──────────────────────────────────────────────────────────

const PHASE1_SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

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

const PHASE2_SYSTEM_PROMPT = `You are a bot detection expert. This user was flagged as a likely bot by an initial screening. Re-evaluate this account carefully and respond with ONLY a number from 0 to 100 representing your confidence that this is a bot (0 = definitely human, 100 = definitely bot).`;

// ── Helpers ──────────────────────────────────────────────────────────

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
    .map((p) => `[${p.created_at}] ${p.text}`)
    .join("\n");

  return `${profile}\n\n${postTexts}`;
}

/** Phase 1: classify a user as BOT or HUMAN. */
async function classifyUser(user: User, datasets: Dataset[]): Promise<{ user: User; isBot: boolean }> {
  const posts = getPostsByUserFromLoaded(user.id, datasets);
  const prompt = buildUserPrompt(user, posts);
  const answer = await chatWithSystem(PHASE1_SYSTEM_PROMPT, prompt, PHASE1_MODEL);
  const isBot = answer.trim().toUpperCase().startsWith("BOT");
  return { user, isBot };
}

/** Phase 2: score confidence (0-100) that a suspected bot is actually a bot. */
async function scoreConfidence(user: User, datasets: Dataset[], model: string): Promise<{ user: User; confidence: number }> {
  const posts = getPostsByUserFromLoaded(user.id, datasets);
  const prompt = buildUserPrompt(user, posts);
  const answer = await chatWithSystem(PHASE2_SYSTEM_PROMPT, prompt, model);
  const parsed = parseInt(answer.trim(), 10);
  const confidence = Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 50;
  return { user, confidence };
}

// ── Main detector ────────────────────────────────────────────────────

export type DetectorResult = {
  runFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

export async function runDetector(
  datasetIds: number[],
  model = MODEL_GEMINI,
  onProgress?: (done: number, total: number) => void,
  delayMs = 50,
  cacheFile?: string,
): Promise<DetectorResult> {
  const paths = datasetPaths(datasetIds);
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

  // Total progress = phase 1 (all users) + phase 2 (suspected bots, unknown yet so we update later)
  const totalPhase1 = users.length;

  // ── Phase 1: classify all users with gemini-3-flash-preview ──

  let cachePath: string;
  let cacheData: CacheData;

  if (cacheFile) {
    cacheData = await loadCache(cacheFile);
    cachePath = cacheFile;
  } else {
    cachePath = await createCache("stats-based", model, datasetIds, users.length);
    cacheData = await loadCache(cachePath);
  }

  const doneIds = new Set(Object.keys(cacheData.results));
  const toClassify = users.filter((u) => !doneIds.has(u.id));
  let doneCount = doneIds.size;

  onProgress?.(doneCount, totalPhase1);

  await Promise.all(
    toClassify.map(async (u, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const result = await classifyUser(u, datasets);
      cacheData.results[result.user.id] = { isBot: result.isBot };
      await writeResult(cachePath, cacheData);
      doneCount++;
      onProgress?.(doneCount, totalPhase1);
      return result;
    }),
  );

  // Collect phase 1 suspected bots
  const phase1Results = users.map((u) => ({
    user: u,
    isBot: cacheData.results[u.id]?.isBot ?? false,
  }));
  const suspectedBots = phase1Results.filter((r) => r.isBot).map((r) => r.user);

  // ── Compute n: how many suspected bots to reclassify as human ──

  const n = Math.round(FALSE_DISCOVERY_RATE * suspectedBots.length);

  if (n === 0 || suspectedBots.length === 0) {
    // No reclassification needed — just output phase 1 results
    const bots = phase1Results.filter((r) => r.isBot);
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const runFile = `${RUNS_DIR}/stats-based-${datasetIds.join("_")}-${timestamp}.txt`;
    const lines = [
      `Datasets: ${datasetIds.join(", ")}`,
      `Detector: stats-based`,
      `Model: ${model}`,
      `Batch size: 1`,
      "",
      ...bots.map((r) => r.user.id),
    ];
    await Bun.write(runFile, lines.join("\n") + "\n");
    return {
      runFile,
      totalUsers: users.length,
      botsDetected: bots.length,
      humansDetected: users.length - bots.length,
    };
  }

  // ── Phase 2: score confidence on suspected bots ──

  const totalOverall = totalPhase1 + suspectedBots.length;
  let phase2Done = 0;
  onProgress?.(totalPhase1, totalOverall);

  const scored: { user: User; confidence: number }[] = await Promise.all(
    suspectedBots.map(async (u, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const result = await scoreConfidence(u, datasets, model);
      phase2Done++;
      onProgress?.(totalPhase1 + phase2Done, totalOverall);
      return result;
    }),
  );

  // Sort by confidence ascending — lowest confidence = most likely human
  scored.sort((a, b) => a.confidence - b.confidence);

  // Reclassify the bottom n as human
  const reclassifiedIds = new Set(scored.slice(0, n).map((s) => s.user.id));

  const finalResults = users.map((u) => {
    const phase1Bot = cacheData.results[u.id]?.isBot ?? false;
    const isBot = phase1Bot && !reclassifiedIds.has(u.id);
    return { user: u, isBot };
  });
  const bots = finalResults.filter((r) => r.isBot);

  // ── Write run file ──

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `${RUNS_DIR}/stats-based-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: stats-based`,
    `Model: ${model}`,
    `Batch size: 1`,
    "",
    ...bots.map((r) => r.user.id),
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

  let model = MODEL_GEMINI;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_GEMINI;
    args.splice(modelIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/stats-based.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Stats-Based Detector | Phase 1: ${PHASE1_MODEL} | Phase 2: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`);
  console.log(`False discovery rate: ${(FALSE_DISCOVERY_RATE * 100).toFixed(1)}%`);

  const result = await runDetector(datasetIds, model, (done, total) => {
    const phase = done <= datasetIds.length ? "Phase 1" : "Phase 2";
    process.stdout.write(`\r${phase}: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
