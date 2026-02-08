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
import { MODEL_ANTHROPIC, type DetectorResult } from "./baseline.ts";

export const MODEL_STAGE1 = "groq/openai/gpt-oss-120b";
export const MODEL_STAGE2 = MODEL_ANTHROPIC;
export const MODEL = MODEL_STAGE1; // default shown in CLI menu

// ── Stage 1: signal extraction (Groq) ────────────────────────────────

const STAGE1_SYSTEM = `You are an expert at detecting bot accounts on social media. You will receive a user's profile and their posts. Evaluate the account on exactly three signals.

IMPORTANT: Set a HIGH bar for YES. Most real humans are opinionated, emotional, and make typos — that is NORMAL. Only answer YES when the pattern is so extreme and mechanical that it is clearly not a real person.

a) BLINDLY_POLITICAL: Is this account EXCLUSIVELY a political propaganda machine? Answer YES only if the account does nothing but push a single political narrative with zero personal content, no humor, no off-topic posts, and reads like an automated campaign. Having strong political opinions, even extreme ones, is normal human behavior — that alone is NOT enough for YES.

b) RAGEBAIT: Is this account systematically manufacturing outrage? Answer YES only if the account's SOLE purpose appears to be provoking anger — every post is inflammatory with no genuine conversation, personal anecdotes, or varied topics. Real humans get angry and post hot takes — that is normal. YES means nearly every post is engineered to provoke.

c) GRAMMAR_MISTAKES: Does this account exhibit MACHINE-LIKE grammar errors? Answer YES only if the errors are clearly non-human — inconsistent language ability (perfect grammar in one sentence, broken in the next), systematic patterns like repeated word substitutions, or text that reads like bad machine translation. Normal human typos, slang, abbreviations, and informal writing are NOT grammar mistakes. Non-native speakers making consistent errors is NOT a bot signal.

Respond with EXACTLY three lines, one per signal, in this format:
BLINDLY_POLITICAL: YES or NO
RAGEBAIT: YES or NO
GRAMMAR_MISTAKES: YES or NO

Nothing else.`;

// ── Stage 2: final classification (Anthropic) ────────────────────────

const STAGE2_SYSTEM = `You are a bot detection expert. You will be given a social media user's profile, their posts, AND the results of a preliminary signal analysis.

YOUR DEFAULT ANSWER SHOULD BE "HUMAN". Most accounts are real people. Only classify as BOT when you are confident based on MULTIPLE strong indicators.

The preliminary signals (BLINDLY_POLITICAL, RAGEBAIT, GRAMMAR_MISTAKES) are hints, not proof. A single YES signal is weak evidence. Even two YES signals can describe a real person who is politically passionate and emotional. Treat the signals as one input among many.

Strong BOT indicators (need multiple):
- Mechanical, repetitive posting with nearly identical structure across tweets
- Posting at impossibly regular intervals (e.g. exactly every 30 minutes)
- Zero personal content — no opinions, anecdotes, humor, or casual conversation
- Content that reads like it was generated or translated by a machine
- Profile that looks auto-generated (random username, stock-photo-like name)

Things that are NORMAL for humans (do NOT use these alone to classify as BOT):
- Strong political opinions, even extreme ones
- Emotional or angry posts
- Typos, slang, abbreviations, informal writing
- Posting a lot about one topic they care about (sports, politics, celebrities)
- Non-native speaker grammar patterns

When in doubt, answer HUMAN.

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

  // Classify users — stagger launches by delayMs to avoid rate limits
  let done = 0;
  const results = await Promise.all(
    users.map(async (u, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const result = await classifyUser(u, datasets, MODEL_STAGE1, MODEL_STAGE2);
      done++;
      onProgress?.(done, users.length);
      return result;
    }),
  );

  const bots = results.filter((r) => r.isBot);

  // Write run file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runFile = `runs/prg-${datasetIds.join("_")}-${timestamp}.txt`;
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
  let delayMs = 50;
  const delayIdx = args.indexOf("--delay");
  if (delayIdx !== -1) {
    delayMs = parseInt(args[delayIdx + 1] ?? "50", 10);
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
