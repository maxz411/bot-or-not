/**
 * Final competition detector — identical to V5 but outputs in submission format.
 *
 * Output: maxilillian.detections.<lang>.txt  (one bot user ID per line)
 *
 * Standalone:  bun run js/detectors/final.ts [--model <model>] [--lang en|fr] [--delay <ms>] <dataset_ids...>
 * Imported:    import { runDetector } from "./detectors/final.ts"
 */

import { chatWithSystem } from "../llm.ts";
import {
  loadDatasets,
  getPostsByUserFromLoaded,
  type User,
  type Post,
  type Dataset,
} from "../data.ts";
import {
  createCache,
  loadCache,
  writeResult,
  type CacheData,
} from "./cache.ts";
import { MODEL_OPENAI_V5 } from "../models.ts";

const TEAM_NAME = "maxilillian";

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

  const sortedPosts = [...posts].sort((a, b) =>
    a.created_at.localeCompare(b.created_at),
  );
  const postLines =
    sortedPosts.length > 0
      ? sortedPosts
          .map(
            (p) =>
              `[${p.created_at}] [id:${p.id}] [lang:${p.lang}] ${p.text}`,
          )
          .join("\n")
      : "(no posts)";

  return `${profile}\n\nPosts:\n${postLines}`;
}

async function classifyUser(
  user: User,
  datasets: Dataset[],
  model: string,
): Promise<{ user: User; isBot: boolean }> {
  const posts = getPostsByUserFromLoaded(user.id, datasets);
  const prompt = buildUserPrompt(user, posts);
  const answer = await chatWithSystem(SYSTEM_PROMPT, prompt, model);
  const isBot = answer.trim().toUpperCase().startsWith("BOT");
  return { user, isBot };
}

export type DetectorResult = {
  runFile: string;
  submissionFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

export async function runDetector(
  datasetIds: number[],
  model = MODEL_OPENAI_V5,
  onProgress?: (done: number, total: number) => void,
  delayMs = 50,
  cacheFile?: string,
  lang = "en",
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
    cachePath = await createCache("final", model, datasetIds, users.length);
    cacheData = await loadCache(cachePath);
  }

  const doneIds = new Set(Object.keys(cacheData.results));
  const toClassify = users.filter((u) => !doneIds.has(u.id));
  let doneCount = doneIds.size;

  onProgress?.(doneCount, users.length);

  // Process with bounded concurrency to get steady progress updates.
  const CONCURRENCY = 20;
  let idx = 0;
  async function worker() {
    while (idx < toClassify.length) {
      const myIdx = idx++;
      const u = toClassify[myIdx]!;
      if (delayMs > 0 && myIdx > 0) await Bun.sleep(delayMs);
      const result = await classifyUser(u, datasets, model);
      cacheData.results[result.user.id] = { isBot: result.isBot };
      await writeResult(cachePath, cacheData);
      doneCount++;
      onProgress?.(doneCount, users.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, toClassify.length) }, () => worker()));

  const results = users.map((u) => ({
    user: u,
    isBot: cacheData.results[u.id]?.isBot ?? false,
  }));
  const bots = results.filter((r) => r.isBot);

  // ── Submission file: just bot IDs, one per line ──
  const submissionFile = `${TEAM_NAME}.detections.${lang}.txt`;
  await Bun.write(submissionFile, bots.map((r) => r.user.id).join("\n") + "\n");

  // ── Also write a standard run file for analysis ──
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `runs/final-${lang}-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: final`,
    `Model: ${model}`,
    `Batch size: 1`,
    "",
    ...bots.map((r) => r.user.id),
  ];
  await Bun.write(runFile, lines.join("\n") + "\n");

  return {
    runFile,
    submissionFile,
    totalUsers: users.length,
    botsDetected: bots.length,
    humansDetected: users.length - bots.length,
  };
}

if (import.meta.main) {
  const args = Bun.argv.slice(2);

  let delayMs = 50;
  const delayIdx = args.indexOf("--delay");
  if (delayIdx !== -1) {
    delayMs = parseInt(args[delayIdx + 1] ?? "50", 10);
    args.splice(delayIdx, 2);
  }

  let model = MODEL_OPENAI_V5;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_OPENAI_V5;
    args.splice(modelIdx, 2);
  }

  let lang = "en";
  const langIdx = args.indexOf("--lang");
  if (langIdx !== -1) {
    lang = args[langIdx + 1] ?? "en";
    args.splice(langIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run js/detectors/final.ts [--model <model>] [--lang en|fr] [--delay <ms>] <dataset_ids...>");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(
    `Final Detector | Team: ${TEAM_NAME} | Lang: ${lang} | Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`,
  );

  const result = await runDetector(
    datasetIds,
    model,
    (done, total) => {
      process.stdout.write(`\rClassifying: ${done}/${total}`);
    },
    delayMs,
    undefined,
    lang,
  );

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Submission file: ${result.submissionFile}`);
  console.log(`Run file: ${result.runFile}`);
}
