/**
 * V2 bot detector — pre-computed features + chain-of-thought + few-shot examples.
 *
 * Improvements over baseline:
 *   1. Statistical features (hashtags, emojis, post length, pipe-in-bio, etc.) injected into prompt
 *   2. Chain-of-thought reasoning before VERDICT
 *   3. Few-shot examples of tricky bots and humans
 *
 * Can be run standalone:  bun run detectors/v2.ts [--model <model>] [--delay <ms>] <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/v2.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { MODEL_DEEPSEEK } from "../models.ts";
import { datasetPaths, RUNS_DIR } from "../paths.ts";

// ── Feature computation ─────────────────────────────────────────────

const EMOJI_REGEX = /[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F000}-\u{1F02F}\u{1F0A0}-\u{1F0FF}]/gu;

function countHashtags(text: string): number {
  return (text.match(/#\w+/g) ?? []).length;
}

function countEmojis(text: string): number {
  return (text.match(EMOJI_REGEX) ?? []).length;
}

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/).filter(Boolean);
}

function stdDev(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
}

function usernameNameSimilarity(username: string, name: string): number {
  if (!username || !name) return 0;
  const u = username.toLowerCase().replace(/[^a-z0-9]/g, "");
  const n = name.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (u.length === 0 || n.length === 0) return 0;
  const uBase = u.replace(/\d+$/, "");
  if (uBase.length >= 3 && (n.includes(uBase) || uBase.includes(n.slice(0, Math.min(n.length, 4))))) return 1;
  // Jaccard on character bigrams
  const bigrams = (s: string) => {
    const set = new Set<string>();
    for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
    return set;
  };
  const a = bigrams(u);
  const b = bigrams(n);
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}

type FeatureVector = {
  hashtagsPerPost: number;
  emojisPerPost: number;
  avgPostLength: number;
  postLengthStdDev: number;
  uniqueWordRatio: number;
  postsPerDay: number;
  descriptionHasPipe: boolean;
  usernameNameSim: number;
  topicConsistency: number;
};

function computeFeatures(user: User, posts: Post[]): FeatureVector {
  const n = posts.length;
  const lengths = posts.map((p) => p.text.length);
  const totalHashtags = posts.reduce((s, p) => s + countHashtags(p.text), 0);
  const totalEmojis = posts.reduce((s, p) => s + countEmojis(p.text), 0);
  const allWords = tokenize(posts.map((p) => p.text).join(" "));
  const uniqueWords = new Set(allWords);
  const days = new Set(posts.map((p) => p.created_at.slice(0, 10))).size;

  const desc = user.description || "";
  const descWords = new Set(tokenize(desc).filter((w) => w.length > 2));
  const overlap = descWords.size > 0 ? allWords.filter((w) => descWords.has(w)).length : 0;

  return {
    hashtagsPerPost: n > 0 ? totalHashtags / n : 0,
    emojisPerPost: n > 0 ? totalEmojis / n : 0,
    avgPostLength: lengths.length > 0 ? lengths.reduce((a, b) => a + b, 0) / lengths.length : 0,
    postLengthStdDev: stdDev(lengths),
    uniqueWordRatio: allWords.length > 0 ? uniqueWords.size / allWords.length : 0,
    postsPerDay: days > 0 ? n / days : 0,
    descriptionHasPipe: /\w\s*\|\s*\w/.test(desc),
    usernameNameSim: usernameNameSimilarity(user.username, user.name),
    topicConsistency: allWords.length > 0 ? overlap / allWords.length : 0,
  };
}

function formatFeatures(f: FeatureVector): string {
  return [
    `Hashtags per post (avg): ${f.hashtagsPerPost.toFixed(2)}`,
    `Emojis per post (avg): ${f.emojisPerPost.toFixed(2)}`,
    `Avg post length (chars): ${f.avgPostLength.toFixed(0)}`,
    `Post length std dev: ${f.postLengthStdDev.toFixed(0)}`,
    `Unique word ratio: ${f.uniqueWordRatio.toFixed(2)}`,
    `Posts per active day: ${f.postsPerDay.toFixed(1)}`,
    `Bio uses "X | Y" pattern: ${f.descriptionHasPipe ? "YES" : "no"}`,
    `Username-name similarity (0-1): ${f.usernameNameSim.toFixed(2)}`,
    `Topic consistency (bio vs posts, 0-1): ${f.topicConsistency.toFixed(3)}`,
  ].join("\n");
}

// ── System prompt ────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a bot detection expert. You will be given a social media user's profile, their posts, and pre-computed statistical signals. Determine if this account is a bot or a real human.

CALIBRATION: Wrongly flagging a real human as a bot is TWICE as costly as missing a bot. Be confident before labeling someone BOT. When in doubt, lean toward HUMAN.

BOT RED FLAGS — any of these suggest BOT:
- Username does not match display name (e.g. username "codeGenie905" but name "Samuel Johnson")
- Bio uses "Name | hobby" pipe-separated format (e.g. "Claire | obsessed with botanical illustration")
- Posts don't match the bio's claimed interests (claims to be a coder but posts about sports/birds/anxiety)
- Formulaic posts: every post has hashtags + emojis in the same structure
- Relentlessly positive/curated tone — no frustration, complaints, or messy human moments
- Uniform post lengths (low std dev) — real humans vary a lot
- Generic cliche bio ("Lover of life, adventure seeker, and coffee enthusiast")
- High tweet count with suspiciously consistent style across all posts

HUMAN INDICATORS — these suggest HUMAN:
- Genuine emotional reactions (anger, frustration, sarcasm, profanity)
- Inconsistent formatting (some posts short, some long, some with typos)
- Specific personal references (real teams, real events, real people they know)
- Posts that feel like genuine conversation replies, not standalone statements

Z-score: how far this user's posting activity deviates from the dataset average (0 = average).

Write a brief analysis (2-3 sentences), then on the final line write ONLY: VERDICT: BOT or VERDICT: HUMAN`;

// ── Prompt builder ──────────────────────────────────────────────────

function buildUserPrompt(user: User, posts: Post[], features: FeatureVector): string {
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

  const signals = formatFeatures(features);

  return `USER PROFILE:\n${profile}\n\nCOMPUTED SIGNALS:\n${signals}\n\nPOSTS (${posts.length} total, showing up to 50):\n${postTexts}`;
}

// ── Classification ──────────────────────────────────────────────────

async function classifyUser(
  user: User,
  datasets: Dataset[],
  model: string,
): Promise<{ user: User; isBot: boolean }> {
  const posts = getPostsByUserFromLoaded(user.id, datasets);
  const features = computeFeatures(user, posts);
  const prompt = buildUserPrompt(user, posts, features);
  const answer = await chatWithSystem(SYSTEM_PROMPT, prompt, model);

  // Parse VERDICT from the last lines (search from bottom)
  const lines = answer.trim().split("\n");
  let isBot = false;
  for (let i = lines.length - 1; i >= Math.max(0, lines.length - 5); i--) {
    const line = lines[i]!.toUpperCase();
    if (line.includes("VERDICT")) {
      // Extract just the part after "VERDICT:"
      const afterVerdict = line.slice(line.indexOf("VERDICT") + 7);
      isBot = afterVerdict.includes("BOT");
      break;
    }
  }
  // Fallback: if no VERDICT found, check if the response starts/ends with BOT
  if (!lines.some((l) => l.toUpperCase().includes("VERDICT"))) {
    const trimmed = answer.trim().toUpperCase();
    isBot = trimmed.startsWith("BOT") || trimmed.endsWith("BOT");
  }

  return { user, isBot };
}

// ── Types ────────────────────────────────────────────────────────────

export type DetectorResult = {
  runFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

// ── Main detector ────────────────────────────────────────────────────

export async function runDetector(
  datasetIds: number[],
  model = MODEL_DEEPSEEK,
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
    cachePath = await createCache("v2", model, datasetIds, users.length);
    cacheData = await loadCache(cachePath);
  }

  const doneIds = new Set(Object.keys(cacheData.results));
  const toClassify = users.filter((u) => !doneIds.has(u.id));
  let doneCount = doneIds.size;

  onProgress?.(doneCount, users.length);

  // Classify only users not yet in cache — stagger by delayMs
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

  // Full results from cache (cached + new)
  const results = users.map((u) => ({
    user: u,
    isBot: cacheData.results[u.id]?.isBot ?? false,
  }));
  const bots = results.filter((r) => r.isBot);

  // Write run file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `${RUNS_DIR}/v2-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: v2`,
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
    console.log("Usage: bun run detectors/v2.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    console.log("Example: bun run detectors/v2.ts --model deepseek/deepseek-v3.2-speciale --delay 100 30");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`V2 Detector | Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`);

  const result = await runDetector(datasetIds, model, (done, total) => {
    process.stdout.write(`\rClassifying: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
