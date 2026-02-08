/**
 * Politics / Ragebait / Grammar bot detector — two-stage pipeline, BATCHED.
 *
 * Same logic as politics-ragebait-grammar.ts but processes 5 users per
 * request in both stages, to test whether accuracy degrades with batching.
 *
 * Stage 1 (Groq · openai/gpt-oss-120b):
 *   Evaluate each user on three signals (in batches of 5):
 *     a) Blindly political — parrots talking points without nuance?
 *     b) Ragebait — tries to incite strong emotional reactions?
 *     c) Arbitrary grammar mistakes — makes unnatural/random errors?
 *
 * Stage 2 (Anthropic · claude-haiku-4-5):
 *   Feed the user profiles, posts, AND signal results from Stage 1
 *   into Claude Haiku for the final BOT / HUMAN classification (in batches of 5).
 *
 * Can be run standalone:  bun run detectors/politics-ragebait-grammar-batched.ts <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/politics-ragebait-grammar-batched.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";
import { MODEL_ANTHROPIC, type DetectorResult } from "./baseline.ts";

export const MODEL_STAGE1 = "groq/openai/gpt-oss-120b";
export const MODEL_STAGE2 = MODEL_ANTHROPIC;
export const MODEL = MODEL_STAGE1;

const BATCH_SIZE = 5;

// ── Stage 1: signal extraction (Groq) — batched ──────────────────────

const STAGE1_SYSTEM = `You are an expert at detecting bot accounts on social media. You will receive multiple users' profiles and their posts. For EACH user, evaluate the account on exactly three signals:

a) BLINDLY_POLITICAL: Does this user post political content in a one-sided, unnuanced way — parroting talking points, repeating slogans, or pushing a political agenda without genuine discussion? This is NOT about whether they have political opinions — it's about whether they seem to be a bot programmed to push a narrative.

b) RAGEBAIT: Is this user trying to provoke strong emotional reactions? Look for inflammatory language, divisive framing, outrage-bait, hot takes designed to generate engagement rather than real conversation.

c) GRAMMAR_MISTAKES: Does this user make arbitrary, unnatural grammar mistakes? Bots sometimes produce text with odd errors — random misspellings, broken syntax, or inconsistent quality that doesn't match how real humans write (either fluently or with consistent natural mistakes). Real humans make natural typos; bots make weird ones.

For each user, respond with exactly three lines prefixed by the user's ID, in this format:
USER_ID BLINDLY_POLITICAL: YES or NO
USER_ID RAGEBAIT: YES or NO
USER_ID GRAMMAR_MISTAKES: YES or NO

Output the lines for all users, nothing else.`;

// ── Stage 2: final classification (Anthropic) — batched ──────────────

const STAGE2_SYSTEM = `You are a bot detection expert. You will be given multiple social media users' profiles, their posts, AND the results of a preliminary signal analysis for each user that assessed whether they are:
- Blindly political (one-sided political posting without nuance)
- Ragebait (trying to provoke strong emotional reactions)
- Making arbitrary grammar mistakes (unnatural errors typical of bots)

Use these signals as strong additional evidence alongside your own analysis of the profiles and posts. Consider:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus

For each user, respond with exactly one line in the format:
USER_ID: BOT
or
USER_ID: HUMAN

Output one line per user, nothing else.`;

// ── Helpers ──────────────────────────────────────────────────────────

type SignalResult = {
  blindlyPolitical: boolean;
  ragebait: boolean;
  grammarMistakes: boolean;
};

function buildUserBlock(user: User, posts: Post[]): string {
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

function buildBatchPrompt(userDataList: { user: User; posts: Post[] }[]): string {
  return userDataList
    .map(({ user, posts }, i) => {
      const block = buildUserBlock(user, posts);
      return `=== USER ${i + 1} (ID: ${user.id}) ===\n${block}`;
    })
    .join("\n\n");
}

function formatSignals(signals: SignalResult): string {
  return [
    `BLINDLY_POLITICAL: ${signals.blindlyPolitical ? "YES" : "NO"}`,
    `RAGEBAIT: ${signals.ragebait ? "YES" : "NO"}`,
    `GRAMMAR_MISTAKES: ${signals.grammarMistakes ? "YES" : "NO"}`,
  ].join("\n");
}

// ── Stage 1 parsing ──────────────────────────────────────────────────

function parseStage1Response(response: string, users: User[]): Map<string, SignalResult> {
  const results = new Map<string, SignalResult>();
  const lines = response.trim().toUpperCase().split("\n");

  // Initialize defaults
  for (const u of users) {
    results.set(u.id, { blindlyPolitical: false, ragebait: false, grammarMistakes: false });
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    for (const user of users) {
      if (trimmed.startsWith(user.id)) {
        const existing = results.get(user.id)!;
        if (trimmed.includes("BLINDLY_POLITICAL") && trimmed.includes("YES")) {
          existing.blindlyPolitical = true;
        }
        if (trimmed.includes("RAGEBAIT") && trimmed.includes("YES")) {
          existing.ragebait = true;
        }
        if (trimmed.includes("GRAMMAR_MISTAKES") && trimmed.includes("YES")) {
          existing.grammarMistakes = true;
        }
      }
    }
  }

  return results;
}

// ── Stage 2 parsing ──────────────────────────────────────────────────

function parseStage2Response(response: string, users: User[]): Map<string, boolean> {
  const results = new Map<string, boolean>();
  const lines = response.trim().split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const match = trimmed.match(/^(\S+)\s*:\s*(BOT|HUMAN)/i);
    if (match) {
      results.set(match[1]!, match[2]!.toUpperCase() === "BOT");
    }
  }

  // Fallback: match by position if ID parsing failed for some
  if (results.size < users.length) {
    const botHumanLines = lines.filter((l) => /BOT|HUMAN/i.test(l));
    for (let i = 0; i < users.length; i++) {
      if (!results.has(users[i]!.id) && botHumanLines[i]) {
        results.set(users[i]!.id, botHumanLines[i]!.toUpperCase().includes("BOT"));
      }
    }
  }

  return results;
}

// ── Batch classification ─────────────────────────────────────────────

async function classifyBatch(
  users: User[],
  datasets: Dataset[],
): Promise<{ user: User; isBot: boolean }[]> {
  const userDataList = users.map((user) => ({
    user,
    posts: getPostsByUserFromLoaded(user.id, datasets),
  }));

  // Stage 1: extract signals for all users in batch
  const stage1Prompt = buildBatchPrompt(userDataList);
  const stage1Answer = await chatWithSystem(STAGE1_SYSTEM, stage1Prompt, MODEL_STAGE1);
  const signalsMap = parseStage1Response(stage1Answer, users);

  // Stage 2: final classification with signals included
  const stage2Blocks = userDataList
    .map(({ user, posts }, i) => {
      const block = buildUserBlock(user, posts);
      const signals = signalsMap.get(user.id)!;
      return `=== USER ${i + 1} (ID: ${user.id}) ===\n${block}\n\n--- PRELIMINARY SIGNAL ANALYSIS ---\n${formatSignals(signals)}`;
    })
    .join("\n\n");

  const stage2Answer = await chatWithSystem(STAGE2_SYSTEM, stage2Blocks, MODEL_STAGE2);
  const botMap = parseStage2Response(stage2Answer, users);

  return users.map((user) => ({
    user,
    isBot: botMap.get(user.id) ?? false,
  }));
}

// ── Public API ───────────────────────────────────────────────────────

/**
 * Run the batched two-stage PRG detector on the given dataset IDs.
 *
 * `model` is accepted for CLI/menu compatibility but ignored internally —
 * Stage 1 always uses MODEL_STAGE1 (Groq) and Stage 2 always uses MODEL_STAGE2 (Anthropic).
 */
export async function runDetector(
  datasetIds: number[],
  _model = MODEL,
  onProgress?: (done: number, total: number) => void,
  delayMs = 500,
): Promise<DetectorResult> {
  const paths = datasetIds.map((id) => `datasets/dataset.posts&users.${id}.json`);
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

  // Split into batches
  const batches: User[][] = [];
  for (let i = 0; i < users.length; i += BATCH_SIZE) {
    batches.push(users.slice(i, i + BATCH_SIZE));
  }

  // Classify batches — stagger by delayMs to avoid rate limits
  let done = 0;
  const allResults: { user: User; isBot: boolean }[] = [];

  const batchResults = await Promise.all(
    batches.map(async (batch, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const results = await classifyBatch(batch, datasets);
      done += batch.length;
      onProgress?.(done, users.length);
      return results;
    }),
  );

  for (const batch of batchResults) {
    allResults.push(...batch);
  }

  const bots = allResults.filter((r) => r.isBot);

  // Write run file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `runs/prg-batched-${datasetIds.join("_")}-${timestamp}.txt`;
  const lines = [
    `Datasets: ${datasetIds.join(", ")}`,
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
  let delayMs = 500;
  const delayIdx = args.indexOf("--delay");
  if (delayIdx !== -1) {
    delayMs = parseInt(args[delayIdx + 1] ?? "500", 10);
    args.splice(delayIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/politics-ragebait-grammar-batched.ts [--delay <ms>] <dataset_ids...>");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Stage 1: ${MODEL_STAGE1} | Stage 2: ${MODEL_STAGE2} | Batch size: ${BATCH_SIZE}`);
  console.log(`Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms`);

  const result = await runDetector(datasetIds, MODEL, (done, total) => {
    process.stdout.write(`\rClassifying: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
