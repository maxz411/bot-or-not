/**
 * Check accuracy of a bot-detection run file against dataset ground truth.
 * TypeScript port of main.py — no Python dependency needed.
 */

import { resolve, basename } from "path";
import { DATASETS_DIR, RESULTS_DIR, PROJECT_ROOT } from "./paths.ts";

type UserMeta = {
  id: string;
  username?: string;
  name?: string;
  description?: string;
  location?: string;
  tweet_count?: number;
  z_score?: number;
};

type SummaryStats = {
  total: number;
  bots: number;
  humans: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  pctCorrect: number;
  score: number;
  maxScore: number;
  pctMax: number;
};

/** Parse first line like 'Datasets: 30, 31, 32' -> [30, 31, 32]. */
function parseDatasetsLine(line: string): number[] {
  const match = line.match(/Datasets:\s*([\d\s,]+)/i);
  if (!match) return [];
  return match[1]!
    .split(",")
    .map((s) => s.trim())
    .filter((s) => /^\d+$/.test(s))
    .map(Number);
}

type RunFileHeader = {
  datasetIds: number[];
  detector: string;
  model: string;
  batchSize: string;
  recursionDepth: string;
};

/**
 * Parse run file header lines. Headers are key: value pairs at the top of
 * the file, separated from bot IDs by a blank line. Supports both old format
 * (just "Datasets: ..." on line 1) and new format with Detector/Model/Batch size.
 * Returns the parsed header and the remaining lines (bot IDs).
 */
function parseRunFileHeader(lines: string[]): { header: RunFileHeader; botLines: string[] } {
  const header: RunFileHeader = {
    datasetIds: [],
    detector: "(unknown)",
    model: "(unknown)",
    batchSize: "(unknown)",
    recursionDepth: "",
  };

  // Find where headers end: first blank line, or first line that looks like a UUID
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  let headerEnd = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    if (line === "" || uuidPattern.test(line)) {
      headerEnd = i;
      break;
    }
    headerEnd = i + 1;
  }

  // Parse header lines
  for (let i = 0; i < headerEnd; i++) {
    const line = lines[i]!.trim();
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim().toLowerCase();
    const value = line.slice(colonIdx + 1).trim();
    switch (key) {
      case "datasets":
        header.datasetIds = parseDatasetsLine(line);
        break;
      case "detector":
        header.detector = value;
        break;
      case "model":
        header.model = value;
        break;
      case "batch size":
        header.batchSize = value;
        break;
      case "recursion depth":
        header.recursionDepth = value;
        break;
    }
  }

  // Remaining lines are bot IDs (skip any blank separator lines)
  const botLines = lines.slice(headerEnd).filter((l) => l.trim() !== "");

  return { header, botLines };
}

function summarizeStats(
  truthByUser: Map<string, boolean>,
  predictedBots: Set<string>,
): SummaryStats {
  let bots = 0;
  let humans = 0;
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  for (const [uid, isBot] of truthByUser) {
    const predictedBot = predictedBots.has(uid);
    if (isBot) {
      bots += 1;
      if (predictedBot) tp += 1;
      else fn += 1;
    } else {
      humans += 1;
      if (predictedBot) fp += 1;
      else tn += 1;
    }
  }

  const total = bots + humans;
  const pctCorrect = total ? (100.0 * (tp + tn)) / total : 0;
  const score = tp * 4 + fn * -1 + fp * -2;
  const maxScore = bots * 4; // perfect: all bots caught, no FP
  const pctMax = maxScore > 0 ? (100.0 * score) / maxScore : 0;

  return {
    total,
    bots,
    humans,
    tp,
    tn,
    fp,
    fn,
    pctCorrect,
    score,
    maxScore,
    pctMax,
  };
}

/** Build {userId: isBot}, {userId: userMeta}, and per-dataset truth maps. */
async function loadUsersAndBots(
  datasetIds: number[],
): Promise<{
  truth: Map<string, boolean>;
  users: Map<string, UserMeta>;
  perDatasetTruth: Map<number, Map<string, boolean>>;
}> {
  const truth = new Map<string, boolean>();
  const users = new Map<string, UserMeta>();
  const perDatasetTruth = new Map<number, Map<string, boolean>>();

  for (const n of datasetIds) {
    const jsonPath = resolve(DATASETS_DIR, `dataset.posts&users.${n}.json`);
    const jsonFile = Bun.file(jsonPath);
    if (!(await jsonFile.exists())) {
      console.log(`Warning: ${jsonPath} not found, skipping dataset ${n}`);
      continue;
    }
    const data = await jsonFile.json();
    const datasetTruth = new Map<string, boolean>();
    perDatasetTruth.set(n, datasetTruth);

    for (const u of data.users ?? []) {
      const uid: string = u.id;
      truth.set(uid, false);
      users.set(uid, u);
      datasetTruth.set(uid, false);
    }

    const botsPath = resolve(DATASETS_DIR, `dataset.bots.${n}.txt`);
    const botsFile = Bun.file(botsPath);
    if (!(await botsFile.exists())) {
      console.log(
        `Warning: ${botsPath} not found, skipping bot labels for dataset ${n}`,
      );
      continue;
    }
    const botsText = await botsFile.text();
    for (const line of botsText.split("\n")) {
      const uid = line.trim();
      if (uid && datasetTruth.has(uid)) {
        datasetTruth.set(uid, true);
      }
      if (uid && truth.has(uid)) {
        truth.set(uid, true);
      }
    }
  }

  return { truth, users, perDatasetTruth };
}

/** Format a user dict into a readable string (matches Python output). */
function formatUser(u: UserMeta): string {
  return [
    `  ID:          ${u.id ?? "?"}`,
    `  Username:    ${u.username ?? "?"}`,
    `  Name:        ${u.name ?? "?"}`,
    `  Description: ${u.description || "(none)"}`,
    `  Location:    ${u.location || "(none)"}`,
    `  Tweets:      ${u.tweet_count ?? "?"}`,
    `  Z-score:     ${u.z_score ?? "?"}`,
  ].join("\n");
}

/**
 * Analyze a run file against ground truth and return the report lines.
 * Also writes the report to results/.
 */
export async function analyzeRunFile(runFilePath: string): Promise<string[]> {
  const absPath = runFilePath.startsWith("/")
    ? runFilePath
    : resolve(PROJECT_ROOT, runFilePath);

  const file = Bun.file(absPath);
  if (!(await file.exists())) {
    return [`Error: run file not found: ${absPath}`];
  }

  const raw = await file.text();
  const lines = raw
    .split("\n")
    .map((l) => l.trim());

  if (lines.length === 0 || (lines.length === 1 && lines[0] === "")) {
    return ["Error: run file is empty"];
  }

  const { header, botLines } = parseRunFileHeader(lines);
  const datasetIds = header.datasetIds;
  if (datasetIds.length === 0) {
    return [
      "Error: could not parse 'Datasets: 30, 31, ...' from first line",
    ];
  }

  // Ground truth
  const { truth, users, perDatasetTruth } = await loadUsersAndBots(datasetIds);
  if (truth.size === 0) {
    return ["Error: no users loaded from datasets"];
  }

  // Predicted bot IDs
  let predictedBots = new Set<string>();
  for (const line of botLines) {
    const uid = line.trim();
    if (uid) predictedBots.add(uid);
  }

  // Warn about predicted IDs not in any dataset
  const truthKeys = new Set(truth.keys());
  const unknown = new Set([...predictedBots].filter((id) => !truthKeys.has(id)));
  const warnings: string[] = [];
  if (unknown.size > 0) {
    warnings.push(
      `Warning: ${unknown.size} predicted user ID(s) not in datasets (ignored):`,
    );
    const sorted = [...unknown].sort();
    for (const uid of sorted.slice(0, 10)) {
      warnings.push(`  ${uid}`);
    }
    if (unknown.size > 10) {
      warnings.push(`  ... and ${unknown.size - 10} more`);
    }
    // Remove unknown IDs from predictions
    predictedBots = new Set([...predictedBots].filter((id) => truthKeys.has(id)));
  }

  const combined = summarizeStats(truth, predictedBots);
  const total = combined.total;
  const tp = combined.tp;
  const tn = combined.tn;
  const fp = combined.fp;
  const fn = combined.fn;
  const score = combined.score;
  const maxScore = combined.maxScore;
  const pctCorrect = combined.pctCorrect;
  const pctMax = combined.pctMax;
  const pctFp = total ? (100.0 * fp) / total : 0;
  const pctFn = total ? (100.0 * fn) / total : 0;

  const tpIds = [...predictedBots]
    .filter((id) => truth.get(id) === true)
    .sort();
  const fpIds = [...predictedBots]
    .filter((id) => truth.get(id) === false)
    .sort();
  const fnIds = [...truth]
    .filter(([, isBot]) => isBot)
    .map(([uid]) => uid)
    .filter((uid) => !predictedBots.has(uid))
    .sort();

  // Build output
  const out: string[] = [];
  out.push("=".repeat(60));
  out.push("RUN ACCURACY REPORT");
  out.push("=".repeat(60));
  out.push(`Run file:    ${basename(absPath)}`);
  out.push(`Detector:    ${header.detector}`);
  out.push(`Model:       ${header.model}`);
  out.push(`Batch size:  ${header.batchSize}`);
  if (header.recursionDepth) {
    out.push(`Rec. depth:  ${header.recursionDepth}`);
  }
  out.push(`Datasets:    [${datasetIds.join(", ")}]`);
  out.push(
    `Total users: ${total}  (bots: ${combined.bots}, humans: ${combined.humans})`,
  );
  out.push("");
  out.push("DATASET BREAKDOWN:");
  for (const datasetId of datasetIds) {
    const datasetTruth = perDatasetTruth.get(datasetId);
    if (!datasetTruth || datasetTruth.size === 0) {
      out.push(`  Dataset ${datasetId}: no users loaded`);
      continue;
    }
    const stats = summarizeStats(datasetTruth, predictedBots);
    out.push(
      `  Dataset ${datasetId}: users=${stats.total} bots=${stats.bots} humans=${stats.humans} | TP=${stats.tp} TN=${stats.tn} FP=${stats.fp} FN=${stats.fn} | acc=${stats.pctCorrect.toFixed(2)}% | score=${stats.score}/${stats.maxScore} (${stats.pctMax.toFixed(1)}%)`,
    );
  }
  out.push(
    `  Combined: users=${combined.total} bots=${combined.bots} humans=${combined.humans} | TP=${combined.tp} TN=${combined.tn} FP=${combined.fp} FN=${combined.fn} | acc=${combined.pctCorrect.toFixed(2)}% | score=${combined.score}/${combined.maxScore} (${combined.pctMax.toFixed(1)}%)`,
  );
  out.push("");
  out.push("COUNTS (COMBINED):");
  out.push(`  True Positives:    ${tp}`);
  out.push(`  True Negatives:    ${tn}`);
  out.push(`  False Positives:   ${fp}  (${pctFp.toFixed(2)}%)`);
  out.push(`  False Negatives:   ${fn}  (${pctFn.toFixed(2)}%)`);
  out.push(`  Correct:           ${tp + tn}  (${pctCorrect.toFixed(2)}%)`);
  out.push("");
  out.push(`SCORE (+4 TP, -1 FN, -2 FP): ${score} / ${maxScore}  (${pctMax.toFixed(1)}%)`);

  // False Positives detail
  out.push("");
  out.push("-".repeat(60));
  out.push(`FALSE POSITIVES — ${fp} human(s) wrongly flagged as bot`);
  out.push("-".repeat(60));
  if (fpIds.length > 0) {
    for (const uid of fpIds) {
      out.push("");
      out.push(formatUser(users.get(uid) ?? { id: uid }));
    }
  } else {
    out.push("  (none)");
  }

  // False Negatives detail
  out.push("");
  out.push("-".repeat(60));
  out.push(`FALSE NEGATIVES — ${fn} bot(s) missed`);
  out.push("-".repeat(60));
  if (fnIds.length > 0) {
    for (const uid of fnIds) {
      out.push("");
      out.push(formatUser(users.get(uid) ?? { id: uid }));
    }
  } else {
    out.push("  (none)");
  }

  const report = out.join("\n");

  // Write to results/
  await Bun.file(RESULTS_DIR).exists().catch(() => false);
  const { mkdir } = await import("node:fs/promises");
  await mkdir(RESULTS_DIR, { recursive: true });

  const stem = basename(absPath).replace(/\.[^.]+$/, "");
  const resultName = `${stem}.results.txt`;
  const resultPath = resolve(RESULTS_DIR, resultName);
  await Bun.write(resultPath, report + "\n");

  // Combine warnings + report
  const fullOutput = [...warnings, ...out];
  fullOutput.push("");
  fullOutput.push(
    `Results saved to: results/${resultName}`,
  );

  return fullOutput;
}
