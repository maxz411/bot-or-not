/**
 * Politics / Ragebait / Grammar bot detector — two-stage pipeline.
 *
 * Stage 1 (Groq · openai/gpt-oss-120b):
 *   Evaluate each user on three signals:
 *     a) Blindly political — parrots talking points without nuance?
 *     b) Ragebait — tries to incite strong emotional reactions?
 *     c) Arbitrary grammar mistakes — makes unnatural/random errors?
 *
 * Stage 2 (Anthropic · claude-haiku-4-5):
 *   Feed the user profile, posts, AND the signal results from Stage 1
 *   into Claude Haiku for the final BOT / HUMAN classification.
 *
 * Can be run standalone:  bun run detectors/politics-ragebait-grammar.ts <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/politics-ragebait-grammar.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { MODEL_ANTHROPIC } from "../models.ts";
import { type DetectorResult } from "./baseline.ts";
import { createCache, loadCache, writeResult, type CacheData } from "./cache.ts";
import { datasetPaths, RUNS_DIR } from "../paths.ts";

export const MODEL_STAGE1 = "groq/openai/gpt-oss-120b";
export const MODEL_STAGE2 = MODEL_ANTHROPIC;
export const MODEL = MODEL_STAGE1; // default shown in CLI menu

// ── Stage 1: signal extraction (Groq) ────────────────────────────────

const STAGE1_SYSTEM = `You are an expert at detecting bot accounts on social media. You will receive a user's profile and their posts. Evaluate the account on exactly three signals:

a) BLINDLY_POLITICAL: Does this user post political content in a one-sided, unnuanced way — parroting talking points, repeating slogans, or pushing a political agenda without genuine discussion? This is NOT about whether they have political opinions — it's about whether they seem to be a bot programmed to push a narrative.

b) RAGEBAIT: Is this user trying to provoke strong emotional reactions? Look for inflammatory language, divisive framing, outrage-bait, hot takes designed to generate engagement rather than real conversation.

c) GRAMMAR_MISTAKES: Does this user make arbitrary, unnatural grammar mistakes? Bots sometimes produce text with odd errors — random misspellings, broken syntax, or inconsistent quality that doesn't match how real humans write (either fluently or with consistent natural mistakes). Real humans make natural typos; bots make weird ones.

Respond with EXACTLY three lines, one per signal, in this format:
BLINDLY_POLITICAL: YES or NO
RAGEBAIT: YES or NO
GRAMMAR_MISTAKES: YES or NO

Nothing else.`;

// ── Stage 2: final classification (Anthropic) ────────────────────────

const STAGE2_SYSTEM = `You are a bot detection expert. You will be given a social media user's profile, their posts, AND the results of a preliminary signal analysis that assessed whether the user is:
- Blindly political (one-sided political posting without nuance)
- Ragebait (trying to provoke strong emotional reactions)
- Making arbitrary grammar mistakes (unnatural errors typical of bots)

Use these signals as strong additional evidence alongside your own analysis of the profile and posts. Consider:
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
    .slice(0, 50)
    .map((p) => `[${p.created_at}] ${p.text}`)
    .join("\n");

  return `USER PROFILE:\n${profile}\n\nPOSTS (${posts.length} total, showing up to 50):\n${postTexts}`;
}

export type SignalResult = {
  blindlyPolitical: boolean;
  ragebait: boolean;
  grammarMistakes: boolean;
};

function parseSignals(answer: string): SignalResult {
  const lines = answer.trim().toUpperCase().split("\n");
  const get = (key: string) =>
    lines.some((l) => l.includes(key) && l.includes("YES"));
  return {
    blindlyPolitical: get("BLINDLY_POLITICAL"),
    ragebait: get("RAGEBAIT"),
    grammarMistakes: get("GRAMMAR_MISTAKES"),
  };
}

function formatSignals(signals: SignalResult): string {
  return [
    `BLINDLY_POLITICAL: ${signals.blindlyPolitical ? "YES" : "NO"}`,
    `RAGEBAIT: ${signals.ragebait ? "YES" : "NO"}`,
    `GRAMMAR_MISTAKES: ${signals.grammarMistakes ? "YES" : "NO"}`,
  ].join("\n");
}

async function classifyUser(
  user: User,
  datasets: Dataset[],
  stage1Model: string,
  stage2Model: string,
): Promise<{ user: User; signals: SignalResult; isBot: boolean }> {
  const posts = getPostsByUserFromLoaded(user.id, datasets);
  const userPrompt = buildUserPrompt(user, posts);

  // Stage 1: extract signals via Groq
  const stage1Answer = await chatWithSystem(STAGE1_SYSTEM, userPrompt, stage1Model);
  const signals = parseSignals(stage1Answer);

  // Stage 2: final call via Anthropic, with signals included
  const stage2Prompt = `${userPrompt}\n\n--- PRELIMINARY SIGNAL ANALYSIS ---\n${formatSignals(signals)}`;
  const stage2Answer = await chatWithSystem(STAGE2_SYSTEM, stage2Prompt, stage2Model);
  const isBot = stage2Answer.trim().toUpperCase().startsWith("BOT");

  return { user, signals, isBot };
}

/**
 * Run the two-stage politics/ragebait/grammar detector on the given dataset IDs.
 *
 * `model` is accepted for CLI/menu compatibility but ignored internally —
 * Stage 1 always uses MODEL_STAGE1 (Groq) and Stage 2 always uses MODEL_STAGE2 (Anthropic).
 * cacheFile: if provided, load and resume from this cache; only missing users are classified.
 */
export async function runDetector(
  datasetIds: number[],
  _model = MODEL,
  onProgress?: (done: number, total: number) => void,
  delayMs = 500,
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
    cachePath = await createCache("prg", _model, datasetIds, users.length);
    cacheData = await loadCache(cachePath);
  }

  const doneIds = new Set(Object.keys(cacheData.results));
  const toClassify = users.filter((u) => !doneIds.has(u.id));
  let doneCount = doneIds.size;

  // Emit initial progress so the UI shows totals immediately (important for resume)
  onProgress?.(doneCount, users.length);

  const newResults = await Promise.all(
    toClassify.map(async (u, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const result = await classifyUser(u, datasets, MODEL_STAGE1, MODEL_STAGE2);
      cacheData.results[result.user.id] = { isBot: result.isBot };
      await writeResult(cachePath, cacheData);
      doneCount++;
      onProgress?.(doneCount, users.length);
      return result;
    }),
  );

  const bots = users.filter((u) => cacheData.results[u.id]?.isBot);

  // Write run file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `${RUNS_DIR}/prg-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
    `Detector: prg (2-stage)`,
    `Model: ${MODEL_STAGE1} → ${MODEL_STAGE2}`,
    `Batch size: 1`,
    "",
    ...bots.map((u) => u.id),
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
  let delayMs = 500;
  const delayIdx = args.indexOf("--delay");
  if (delayIdx !== -1) {
    delayMs = parseInt(args[delayIdx + 1] ?? "500", 10);
    args.splice(delayIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/politics-ragebait-grammar.ts [--delay <ms>] <dataset_ids...>");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Stage 1: ${MODEL_STAGE1} | Stage 2: ${MODEL_STAGE2}`);
  console.log(`Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`);

  const result = await runDetector(datasetIds, MODEL, (done, total) => {
    process.stdout.write(`\rClassifying: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
