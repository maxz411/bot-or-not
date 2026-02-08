/**
 * Batching bot detector — asks an LLM to classify 5 users at a time.
 *
 * Same logic as baseline.ts but sends batches of 5 users per request
 * instead of 1, to test whether accuracy degrades with batching.
 *
 * Can be run standalone:  bun run detectors/batching.ts <dataset_ids...>
 * Or imported:            import { runDetector } from "./detectors/batching.ts"
 */

import { chatWithSystem } from "../llm.ts";
import { loadDatasets, getPostsByUserFromLoaded, type User, type Post, type Dataset } from "../data.ts";

export const MODEL_OPENAI = "openai/gpt-5.1";
export const MODEL_ANTHROPIC = "anthropic/claude-haiku-4-5";

const BATCH_SIZE = 5;

const SYSTEM_PROMPT = `You are a bot detection expert. You will be given multiple social media users' profiles and their posts. Your job is to determine if each account is a bot or a real human.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus

For each user, respond with exactly one line in the format:
USER_ID: BOT
or
USER_ID: HUMAN

where USER_ID is the user's ID. Output one line per user, nothing else.`;

function buildBatchPrompt(userDataList: { user: User; posts: Post[] }[]): string {
  return userDataList
    .map(({ user, posts }, i) => {
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

      return `=== USER ${i + 1} (ID: ${user.id}) ===\nUSER PROFILE:\n${profile}\n\nPOSTS (${posts.length} total, showing up to 50):\n${postTexts}`;
    })
    .join("\n\n");
}

function parseResponse(response: string, users: User[]): Map<string, boolean> {
  const results = new Map<string, boolean>();
  const lines = response.trim().split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Match "USER_ID: BOT" or "USER_ID: HUMAN"
    const match = trimmed.match(/^(\S+)\s*:\s*(BOT|HUMAN)/i);
    if (match) {
      const userId = match[1]!;
      const isBot = match[2]!.toUpperCase() === "BOT";
      results.set(userId, isBot);
    }
  }

  // Fallback: if parsing failed for some users, try to match by position
  if (results.size < users.length) {
    const botHumanLines = lines.filter((l) => /BOT|HUMAN/i.test(l));
    for (let i = 0; i < users.length; i++) {
      if (!results.has(users[i]!.id) && botHumanLines[i]) {
        const isBot = botHumanLines[i]!.toUpperCase().includes("BOT");
        results.set(users[i]!.id, isBot);
      }
    }
  }

  return results;
}

async function classifyBatch(
  users: User[],
  datasets: Dataset[],
  model: string,
): Promise<{ user: User; isBot: boolean }[]> {
  const userDataList = users.map((user) => ({
    user,
    posts: getPostsByUserFromLoaded(user.id, datasets),
  }));

  const prompt = buildBatchPrompt(userDataList);
  const answer = await chatWithSystem(SYSTEM_PROMPT, prompt, model);
  const parsed = parseResponse(answer, users);

  return users.map((user) => ({
    user,
    isBot: parsed.get(user.id) ?? false, // default to HUMAN if parse failed
  }));
}

export type DetectorResult = {
  runFile: string;
  totalUsers: number;
  botsDetected: number;
  humansDetected: number;
};

/**
 * Run the batching detector on the given dataset IDs.
 * onProgress is called after each batch is classified (done / total).
 * delayMs adds a delay between each batch request.
 * Returns the path to the run file + summary counts.
 */
export async function runDetector(
  datasetIds: number[],
  model = MODEL_OPENAI,
  onProgress?: (done: number, total: number) => void,
  delayMs = 50,
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

  // Split into batches of BATCH_SIZE
  const batches: User[][] = [];
  for (let i = 0; i < users.length; i += BATCH_SIZE) {
    batches.push(users.slice(i, i + BATCH_SIZE));
  }

  // Classify batches — stagger launches by delayMs to avoid rate limits
  let done = 0;
  const allResults: { user: User; isBot: boolean }[] = [];

  const batchResults = await Promise.all(
    batches.map(async (batch, i) => {
      if (delayMs > 0 && i > 0) await Bun.sleep(delayMs * i);
      const results = await classifyBatch(batch, datasets, model);
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
  const runFile = `runs/batching-${datasetIds.join("_")}-${timestamp}.txt`;
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

  // Parse --model <model> flag
  let model = MODEL_OPENAI;
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1) {
    model = args[modelIdx + 1] ?? MODEL_OPENAI;
    args.splice(modelIdx, 2);
  }

  if (args.length === 0) {
    console.log("Usage: bun run detectors/batching.ts [--model <model>] [--delay <ms>] <dataset_ids...>");
    console.log("Example: bun run detectors/batching.ts --model anthropic/claude-haiku-4-5 --delay 100 30 31");
    process.exit(1);
  }

  const datasetIds = args.map(Number);
  console.log(`Model: ${model} | Datasets: ${datasetIds.join(", ")} | Delay: ${delayMs}ms | Batch size: ${BATCH_SIZE}`);

  const result = await runDetector(datasetIds, model, (done, total) => {
    process.stdout.write(`\rClassifying: ${done}/${total}`);
  }, delayMs);

  console.log(`\n\nDone! ${result.botsDetected} bots, ${result.humansDetected} humans.`);
  console.log(`Run file: ${result.runFile}`);
}
