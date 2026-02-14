/**
 * Baseline bot detector — asks an LLM for each user in parallel.
 *
 * Two variants:
 *   - OpenAI  (gpt-5.1)
 *   - Anthropic (claude-haiku-4-5)
 *
 * Can be run standalone:  bun run detectors/baseline.ts <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/baseline.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { MODEL_OPENAI } from "../models.ts";
import { datasetPaths, RUNS_DIR } from "../paths.ts";

const SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus

Respond with ONLY "BOT" or "HUMAN" — nothing else.`;

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
    .slice(0, 50) // cap at 50 posts to stay within context
    .map((p) => `[${p.created_at}] ${p.text}`)
    .join("\n");

  return `USER PROFILE:\n${profile}\n\nPOSTS (${posts.length} total, showing up to 50):\n${postTexts}`;
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

/**
 * Run the baseline detector on the given dataset IDs.
 * onProgress is called after each user is classified (done / total).
 * delayMs adds a delay between each request (0 = no delay, all parallel).
 * cacheFile: if provided, load and resume from this cache; only missing users are classified.
 * Returns the path to the run file + summary counts.
 */
export async function runDetector(
  datasetIds: number[],
  model = MODEL_OPENAI,
  onProgress?: (done: number, total: number) => void,
  delayMs = 50,
  cacheFile?: string,
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

  let cachePath: string;
  let cacheData: CacheData;

  if (cacheFile) {
    cacheData = await loadCache(cacheFile);
    cachePath = cacheFile;
  } else {
    cachePath = await createCache("baseline", model, datasetIds, users.length);
    cacheData = await loadCache(cachePath);
  }

  const doneIds = new Set(Object.keys(cacheData.results));
  const toClassify = users.filter((u) => !doneIds.has(u.id));
  let doneCount = doneIds.size;

  // Emit initial progress so the UI shows totals immediately (important for resume)
  onProgress?.(doneCount, users.length);

  // Classify only users not yet in cache — stagger by delayMs
  const newResults = await Promise.all(
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

  // Full results from cache (cached + new)
  const results = users.map((u) => ({
    user: u,
    isBot: cacheData.results[u.id]?.isBot ?? false,
  }));
  const bots = results.filter((r) => r.isBot);

  // Write run file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `${RUNS_DIR}/baseline-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: baseline`,
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

  // Parse --delay <ms> flag
  let delayMs = 50;
  const delayIdx = args.indexOf("--delay");
  if (delayIdx !== -1) {
    delayMs = parseInt(args[delayIdx + 1] ?? "50", 10);
    args.splice(delayIdx, 2);
  }

  // Parse --model <model> flag
  let model = MODEL_OPENAI;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_OPENAI;
    args.splice(modelIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/baseline.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    console.log("Example: bun run detectors/baseline.ts --model anthropic/claude-haiku-4-5 --delay 100 30 31");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`);

  const result = await runDetector(datasetIds, model, (done, total) => {
    process.stdout.write(`\rClassifying: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
