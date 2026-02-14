/**
 * V4 bot detector — v3 + scoring incentive.
 *
 * Can be run standalone:  bun run detectors/v4.ts [--model <model>] [--delay <ms>] <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/v4.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { MODEL_DEEPSEEK } from "../models.ts";
import { datasetPaths, RUNS_DIR } from "../paths.ts";

const SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus
- Political
- Inciting anger or hate by being grossly ignorant.
- Bots tend to post in a schedule that would be unrealistic for a human (consider work and sleep)

+4 for a correctly flagged bot, -1 for a missed bot, -2 for a wrongly flagged human.

Respond with ONLY "BOT" or "HUMAN" — nothing else.`;

function buildUserPrompt(user: User, posts: Post[]): string {
  const profile = [
    `User ID: ${user.id}`,
    `Username: ${user.username}`,
    `Name: ${user.name}`,
    `Description: ${user.description || "(none)"}`,
    `Location: ${user.location || "(none)"}`,
    `Tweet count: ${user.tweet_count}`,
    `Z-score (posting activity deviation from average): ${user.z_score.toFixed(4)}`,
  ].join("\n");

  const postTexts = posts
    .map((p) => `[${p.created_at}] [id:${p.id}] [lang:${p.lang}] ${p.text}`)
    .join("\n");

  return `${profile}\n\n${postTexts}`;
}

async function classifyUser(user: User, datasets: Dataset[], model: string): Promise<{ user: User; isBot: boolean }> {
  const posts = getPostsByUserFromLoaded(user.id, datasets);
  const prompt = buildUserPrompt(user, posts);
  const answer = await chatWithSystem(SYSTEM_PROMPT, prompt, model);
  const isBot = answer.trim().toUpperCase().startsWith("BOT");
  return { user, isBot };
}

export type DetectorResult = {
  runFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

export async function runDetector(
  datasetIds: number[],
  model = MODEL_DEEPSEEK,
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

  let cachePath: string;
  let cacheData: CacheData;

  if (cacheFile) {
    cacheData = await loadCache(cacheFile);
    cachePath = cacheFile;
  } else {
    cachePath = await createCache("v4", model, datasetIds, users.length);
    cacheData = await loadCache(cachePath);
  }

  const doneIds = new Set(Object.keys(cacheData.results));
  const toClassify = users.filter((u) => !doneIds.has(u.id));
  let doneCount = doneIds.size;

  onProgress?.(doneCount, users.length);

  await Promise.all(
    toClassify.map(async (u, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const result = await classifyUser(u, datasets, model);
      cacheData.results[result.user.id] = { isBot: result.isBot };
      await writeResult(cachePath, cacheData);
      doneCount++;
      onProgress?.(doneCount, users.length);
      return result;
    }),
  );

  const results = users.map((u) => ({
    user: u,
    isBot: cacheData.results[u.id]?.isBot ?? false,
  }));
  const bots = results.filter((r) => r.isBot);

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `${RUNS_DIR}/v4-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: v4`,
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

  let model = MODEL_DEEPSEEK;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_DEEPSEEK;
    args.splice(modelIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/v4.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`V4 Detector | Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`);

  const result = await runDetector(datasetIds, model, (done, total) => {
    process.stdout.write(`\rClassifying: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
