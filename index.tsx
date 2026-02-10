import React, { useState, useMemo } from "react";
import { render, Box, Text, useInput, useApp } from "ink";
import { chat, generate, stream } from "./llm.ts";
import { analyzeRunFile } from "./analysis.ts";
import { getPostsByUser, getUserMetadata, type Post, type User } from "./data.ts";
import { runDetector as runBaseline } from "./detectors/baseline.ts";
import { MODEL_OPENAI, MODEL_ANTHROPIC, MODEL_ANTHROPIC_SONNET, MODEL_DEEPSEEK, MODEL_GEMINI, MODEL_ANTHROPIC_OPUS, MODEL_GEMINI_PRO, MODEL_GLM, MODEL_KIMI, MODEL_GROK, MODEL_MISTRAL_LARGE } from "./models.ts";
import { runDetector as runBatching } from "./detectors/batching.ts";
import { runDetector as runPRG, MODEL_STAGE1 as PRG_STAGE1, MODEL_STAGE2 as PRG_STAGE2 } from "./detectors/politics-ragebait-grammar.ts";
import { runDetector as runPRGBatched } from "./detectors/politics-ragebait-grammar-batched.ts";
import { runDetector as runRecursive, RECURSION_DEPTH, TOTAL_ROUNDS, type RecursiveProgress, type CompletedRound } from "./detectors/recursive.ts";
import { runDetector as runInvRecursive, RECURSION_DEPTH as INV_RECURSION_DEPTH, TOTAL_ROUNDS as INV_TOTAL_ROUNDS } from "./detectors/inv_recursive.ts";
import { runDetector as runV2 } from "./detectors/v2.ts";
import { runDetector as runV3 } from "./detectors/v3.ts";
import { runDetector as runV4 } from "./detectors/v4.ts";
import { runDetector as runStatsBased } from "./detectors/stats-based.ts";
import { runDetector as runRecursiveV2, RECURSION_DEPTH as RV2_RECURSION_DEPTH, TOTAL_ROUNDS as RV2_TOTAL_ROUNDS } from "./detectors/recursive-v2.ts";
import { listIncompleteCaches, type IncompleteCache } from "./detectors/cache.ts";
import { clearCache } from "./llm-cache.ts";
import { Glob } from "bun";

// ── Types ────────────────────────────────────────────────────────────

type RunDetectorFn = (datasetIds: number[], model: string, onProgress?: (done: number, total: number) => void, delayMs?: number, cacheFile?: string) => Promise<{ runFile: string; totalUsers: number; botsDetected: number; humansDetected: number }>;

function getDetectorFn(detector: string): RunDetectorFn {
  switch (detector) {
    case "v4": return runV4 as RunDetectorFn;
    case "v3": return runV3 as RunDetectorFn;
    case "v2": return runV2 as RunDetectorFn;
    case "recursive-v2": return runRecursiveV2 as RunDetectorFn;
    case "baseline": return runBaseline as RunDetectorFn;
    case "batching": return runBatching as RunDetectorFn;
    case "prg": return runPRG as RunDetectorFn;
    case "prg-batched": return runPRGBatched as RunDetectorFn;
    case "recursive": return runRecursive as RunDetectorFn;
    case "inv_recursive": return runInvRecursive as RunDetectorFn;
    case "stats-based": return runStatsBased as RunDetectorFn;
    default: return runBaseline as RunDetectorFn;
  }
}

// ── Helpers ──────────────────────────────────────────────────────────

async function allDatasetPaths(): Promise<string[]> {
  const glob = new Glob("datasets/dataset.posts&users.*.json");
  const paths: string[] = [];
  for await (const path of glob.scan(".")) paths.push(path);
  return paths.sort();
}

// ── High scores ──────────────────────────────────────────────────────

type HighScoreEntry = {
  file: string;
  detector: string;
  model: string;
  datasets: string;
  score: number;
  maxScore: number;
  pctMax: number;
  accuracy: string;
  tp: number;
  fp: number;
  fn: number;
};

async function getHighScores(limit = 10): Promise<HighScoreEntry[]> {
  const glob = new Glob("results/*.results.txt");
  const entries: HighScoreEntry[] = [];

  for await (const path of glob.scan(".")) {
    try {
      const text = await Bun.file(path).text();
      const lines = text.split("\n");

      let detector = "", model = "", datasets = "", score = 0, accuracy = "";
      let tp = 0, fp = 0, fn = 0, totalBots = 0;

      for (const line of lines) {
        const m = (key: string) => {
          if (line.startsWith(`${key}:`)) return line.slice(key.length + 1).trim();
          return null;
        };
        const v = m("Detector") ?? m("detector");
        if (v) detector = v;
        const mv = m("Model") ?? m("model");
        if (mv) model = mv;
        const dv = m("Datasets") ?? m("datasets");
        if (dv) datasets = dv;

        const botsMatch = line.match(/bots:\s*(\d+)/);
        if (botsMatch) totalBots = parseInt(botsMatch[1]!);

        const tpMatch = line.match(/True Positives:\s+(\d+)/);
        if (tpMatch) tp = parseInt(tpMatch[1]!);
        const fpMatch = line.match(/False Positives:\s+(\d+)/);
        if (fpMatch) fp = parseInt(fpMatch[1]!);
        const fnMatch = line.match(/False Negatives:\s+(\d+)/);
        if (fnMatch) fn = parseInt(fnMatch[1]!);

        const scoreMatch = line.match(/SCORE.*?:\s*(-?\d+)/);
        if (scoreMatch) score = parseInt(scoreMatch[1]!);
        const accMatch = line.match(/Correct:\s+\d+\s+\(([^)]+)\)/);
        if (accMatch) accuracy = accMatch[1]!;
      }

      const maxScore = totalBots * 4;
      const pctMax = maxScore > 0 ? (100.0 * score) / maxScore : 0;

      if (detector || model) {
        entries.push({ file: path.replace("results/", ""), detector, model, datasets, score, maxScore, pctMax, accuracy, tp, fp, fn });
      }
    } catch {
      // skip unreadable files
    }
  }

  entries.sort((a, b) => b.pctMax - a.pctMax);
  return entries.slice(0, limit);
}

// ── Menu data ────────────────────────────────────────────────────────

type DetectorType = "baseline" | "prg" | "recursive" | "inv_recursive" | "v2" | "v3" | "v4" | "recursive-v2" | "stats-based";

type DetectorInfo = {
  name: string;
  type: DetectorType;
  hasBatched: boolean;
  hasModelChoice: boolean;
  fixedModelDisplay: string;
};

const DETECTORS: DetectorInfo[] = [
  { name: "V4", type: "v4", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "V3", type: "v3", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "V2", type: "v2", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "Baseline", type: "baseline", hasBatched: true, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "PRG 2-stage", type: "prg", hasBatched: true, hasModelChoice: false, fixedModelDisplay: `${PRG_STAGE1} → ${PRG_STAGE2}` },
  { name: "Recursive", type: "recursive", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "Recursive V2", type: "recursive-v2", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "Inv Recursive", type: "inv_recursive", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
  { name: "Stats-Based", type: "stats-based", hasBatched: false, hasModelChoice: true, fixedModelDisplay: "" },
];

const MODEL_CHOICES = [
  { name: `OpenAI (${MODEL_OPENAI})`, value: MODEL_OPENAI },
  { name: `Anthropic (${MODEL_ANTHROPIC})`, value: MODEL_ANTHROPIC },
  { name: `Anthropic (${MODEL_ANTHROPIC_SONNET})`, value: MODEL_ANTHROPIC_SONNET },
  { name: `DeepSeek (${MODEL_DEEPSEEK})`, value: MODEL_DEEPSEEK },
  { name: `Gemini (${MODEL_GEMINI})`, value: MODEL_GEMINI },
  { name: `Anthropic (${MODEL_ANTHROPIC_OPUS})`, value: MODEL_ANTHROPIC_OPUS },
  { name: `Gemini Pro (${MODEL_GEMINI_PRO})`, value: MODEL_GEMINI_PRO },
  { name: `GLM (${MODEL_GLM})`, value: MODEL_GLM },
  { name: `Kimi (${MODEL_KIMI})`, value: MODEL_KIMI },
  { name: `Grok (${MODEL_GROK})`, value: MODEL_GROK },
  { name: `Mistral Large (${MODEL_MISTRAL_LARGE})`, value: MODEL_MISTRAL_LARGE },
];

type ListItem = { name: string; action: string; dimDetail?: string };

const MAIN_MENU: ListItem[] = [
  { name: "V4", action: "det-v4", dimDetail: "v3 + scoring incentive" },
  { name: "V3", action: "det-v3", dimDetail: "minimal prompt" },
  { name: "V2", action: "det-v2", dimDetail: "features + CoT + few-shot" },
  { name: "Baseline", action: "det-baseline", dimDetail: "single-prompt classification" },
  { name: "PRG 2-stage", action: "det-prg", dimDetail: `${PRG_STAGE1} → ${PRG_STAGE2}` },
  { name: "Recursive", action: "det-recursive", dimDetail: "iteratively filter humans" },
  { name: "Recursive V2", action: "det-recursive-v2", dimDetail: "recursive + v3 prompt" },
  { name: "Inv Recursive", action: "det-inv_recursive", dimDetail: "iteratively filter bots" },
  { name: "Stats-Based", action: "det-stats-based", dimDetail: "two-phase FP correction" },
  { name: "High Scores", action: "high-scores", dimDetail: "top 10 runs by score" },
  { name: "Regenerate results", action: "regen-results", dimDetail: "re-analyze all run files" },
  { name: "Get user posts", action: "data-posts", dimDetail: "lookup by ID or username" },
  { name: "Get user metadata", action: "data-metadata", dimDetail: "lookup by ID or username" },
  { name: "API Tests", action: "api-tests", dimDetail: "chat, generate, stream" },
  { name: "Resume run", action: "resume" },
  { name: "Clear LLM cache", action: "clear-cache" },
];

const API_MENU: ListItem[] = [
  { name: "OpenAI SDK — chat()", action: "api-chat" },
  { name: "Vercel AI — generate()", action: "api-generate" },
  { name: "Vercel AI — stream()", action: "api-stream" },
  { name: "Get user posts", action: "data-posts" },
  { name: "Get user metadata", action: "data-metadata" },
];

const BATCH_MENU: ListItem[] = [
  { name: "Standard", action: "standard" },
  { name: "Batched", action: "batched" },
];

const AVAILABLE_DATASETS = [30, 31, 32, 33];

// ── Search items (all possible tasks, flattened) ─────────────────────

type SearchItem = {
  name: string;
  detType?: DetectorType;
  batched?: boolean;
  model?: string;
  modelDisplay?: string;
  apiAction?: string;
  needsInput?: boolean;
  specialAction?: string;
};

function buildSearchItems(): SearchItem[] {
  const items: SearchItem[] = [];

  for (const det of DETECTORS) {
    const batchVariants: [string, boolean][] = det.hasBatched
      ? [["", false], ["Batched", true]]
      : [["", false]];

    const modelVariants: [string, string, string][] = det.hasModelChoice
      ? MODEL_CHOICES.map((m) => [m.name.split(" ")[0]!, m.value, m.name])
      : [["", det.fixedModelDisplay, det.fixedModelDisplay]];

    for (const [batchLabel, batched] of batchVariants) {
      for (const [modelShort, modelVal, modelFull] of modelVariants) {
        const parts = [det.name, batchLabel, modelShort].filter(Boolean);
        items.push({
          name: parts.join(" "),
          detType: det.type,
          batched,
          model: modelVal,
          modelDisplay: modelFull,
        });
      }
    }
  }

  items.push({ name: "API: chat()", apiAction: "api-chat" });
  items.push({ name: "API: generate()", apiAction: "api-generate" });
  items.push({ name: "API: stream()", apiAction: "api-stream" });
  items.push({ name: "Get user posts", apiAction: "data-posts", needsInput: true });
  items.push({ name: "Get user metadata", apiAction: "data-metadata", needsInput: true });
  items.push({ name: "High Scores", specialAction: "high-scores" });
  items.push({ name: "Regenerate results", specialAction: "regen-results" });
  items.push({ name: "Resume run", specialAction: "resume" });
  items.push({ name: "Clear LLM cache", specialAction: "clear-cache" });

  return items;
}

const ALL_SEARCH_ITEMS = buildSearchItems();

function filterSearch(query: string): SearchItem[] {
  if (!query.trim()) return ALL_SEARCH_ITEMS;
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return ALL_SEARCH_ITEMS.filter((item) => {
    const name = item.name.toLowerCase();
    return words.every((w) => name.includes(w));
  });
}

// ── Detector setup helper ────────────────────────────────────────────

function resolveDetectorFn(type: DetectorType, batched: boolean): any {
  switch (type) {
    case "v4": return runV4;
    case "v3": return runV3;
    case "v2": return runV2;
    case "recursive-v2": return runRecursiveV2;
    case "baseline": return batched ? runBatching : runBaseline;
    case "prg": return batched ? runPRGBatched : runPRG;
    case "recursive": return runRecursive;
    case "inv_recursive": return runInvRecursive;
    case "stats-based": return runStatsBased;
  }
}

function isRecursiveType(type: DetectorType): boolean {
  return type === "recursive" || type === "inv_recursive" || type === "recursive-v2";
}

// ── Components ───────────────────────────────────────────────────────

type Screen =
  | "menu"
  | "api-tests"
  | "batch-select"
  | "model-select"
  | "dataset-select"
  | "search"
  | "input"
  | "result"
  | "detecting"
  | "recursive-detecting"
  | "resume-select";

function App() {
  const { exit } = useApp();
  const [screen, setScreen] = useState<Screen>("menu");
  const [cursor, setCursor] = useState(0);
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<string[]>([]);

  // Wizard state
  const [selectedDetector, setSelectedDetector] = useState<DetectorInfo | null>(null);
  const [selectedBatched, setSelectedBatched] = useState(false);
  const [detectorModel, setDetectorModel] = useState(MODEL_OPENAI);
  const [detectorModelDisplay, setDetectorModelDisplay] = useState(MODEL_OPENAI);
  const [detectorFn, setDetectorFn] = useState<{ fn: any }>({ fn: runBaseline });

  // Dataset selection
  const [dsCursor, setDsCursor] = useState(0);
  const [dsChecked, setDsChecked] = useState<boolean[]>(AVAILABLE_DATASETS.map(() => false));

  // Progress
  const [progress, setProgress] = useState({ done: 0, total: 0 });

  // Recursive detector state
  const [recursiveProgress, setRecursiveProgress] = useState<RecursiveProgress>({
    currentRound: 0, totalRounds: TOTAL_ROUNDS, roundDone: 0, roundTotal: 0, roundLabel: "", completedRounds: [],
  });
  const [isInvRecursive, setIsInvRecursive] = useState(false);

  // Resume state
  const [resumeCaches, setResumeCaches] = useState<IncompleteCache[]>([]);
  const [resumeCursor, setResumeCursor] = useState(0);
  const [resumeLoading, setResumeLoading] = useState(false);

  // Search state
  const [searchBuf, setSearchBuf] = useState("");
  const [searchCursor, setSearchCursor] = useState(0);
  const [returnScreen, setReturnScreen] = useState<Screen>("menu");
  const searchResults = useMemo(() => filterSearch(searchBuf), [searchBuf]);

  // Input state
  const [inputBuf, setInputBuf] = useState("");
  const [inputLabel, setInputLabel] = useState("");
  const [inputCallback, setInputCallback] = useState<{ fn: (val: string) => void }>({ fn: () => {} });

  // ── Helpers ──

  const resetDatasets = () => {
    setDsCursor(0);
    setDsChecked(AVAILABLE_DATASETS.map(() => false));
  };

  const setupAndGoToDatasets = (type: DetectorType, batched: boolean, model: string, modelDisplay: string) => {
    const det = DETECTORS.find((d) => d.type === type)!;
    setSelectedDetector(det);
    setSelectedBatched(batched);
    setDetectorModel(model);
    setDetectorModelDisplay(modelDisplay);
    setDetectorFn({ fn: resolveDetectorFn(type, batched) });
    setIsInvRecursive(type === "inv_recursive");
    resetDatasets();
    setScreen("dataset-select");
  };

  const openSearch = () => {
    setReturnScreen(screen);
    setSearchBuf("");
    setSearchCursor(0);
    setScreen("search");
  };

  const showInput = (label: string, cb: (val: string) => void) => {
    setInputLabel(label);
    setInputBuf("");
    setInputCallback({ fn: cb });
    setScreen("input");
  };

  // ── Execute search item ──

  const executeSearchItem = (item: SearchItem) => {
    if (item.detType) {
      setupAndGoToDatasets(item.detType, item.batched!, item.model!, item.modelDisplay || item.model!);
    } else if (item.apiAction) {
      if (item.needsInput) {
        if (item.apiAction === "data-posts") {
          showInput("Enter user ID or username", (val) => executeDataPosts(val));
        } else if (item.apiAction === "data-metadata") {
          showInput("Enter user ID or username", (val) => executeDataMetadata(val));
        }
      } else {
        executeApiTest(item.apiAction);
      }
    } else if (item.specialAction === "high-scores") {
      executeHighScores();
    } else if (item.specialAction === "regen-results") {
      executeRegenResults();
    } else if (item.specialAction === "resume") {
      openResumeScreen();
    } else if (item.specialAction === "clear-cache") {
      executeClearCache();
    }
  };

  // ── Actions ──

  const executeClearCache = () => {
    clearCache();
    setOutput(["LLM cache cleared."]);
    setLoading(false);
    setScreen("result");
  };

  const executeHighScores = async () => {
    setLoading(true);
    setOutput([]);
    setScreen("result");
    try {
      const scores = await getHighScores(10);
      if (scores.length === 0) {
        setOutput(["No results found in results/."]);
      } else {
        const lines: string[] = [
          "TOP 10 RUNS BY SCORE (% of max)",
          "=".repeat(60),
          "",
        ];
        for (let i = 0; i < scores.length; i++) {
          const s = scores[i]!;
          lines.push(`  #${i + 1}  Score: ${s.score} / ${s.maxScore}  (${s.pctMax.toFixed(1)}%)  Accuracy: ${s.accuracy}`);
          lines.push(`      Detector: ${s.detector}  Model: ${s.model}`);
          lines.push(`      TP: ${s.tp}  FP: ${s.fp}  FN: ${s.fn}  Datasets: ${s.datasets}`);
          lines.push("");
        }
        setOutput(lines);
      }
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }
    setLoading(false);
  };

  const executeRegenResults = async () => {
    setLoading(true);
    setOutput([]);
    setScreen("result");
    try {
      const glob = new Glob("runs/*.txt");
      const runFiles: string[] = [];
      for await (const path of glob.scan(".")) runFiles.push(path);
      runFiles.sort();

      const lines: string[] = [`Regenerating results for ${runFiles.length} run files...`, ""];
      let count = 0;
      for (const runFile of runFiles) {
        await analyzeRunFile(runFile);
        count++;
      }
      lines.push(`Done! Regenerated ${count} results files.`);
      setOutput(lines);
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }
    setLoading(false);
  };

  const executeApiTest = async (action: string) => {
    setLoading(true);
    setOutput([]);
    setScreen("result");
    try {
      let lines: string[] = [];
      if (action === "api-chat") {
        const res = await chat("Say hi in 5 words.");
        lines = [`Response: ${res}`];
      } else if (action === "api-generate") {
        const res = await generate("Say hi in 5 words.");
        lines = [`Response: ${res}`];
      } else if (action === "api-stream") {
        const result = await stream("Say hi in 5 words.");
        let text = "";
        for await (const chunk of result.textStream) text += chunk;
        lines = [`Response: ${text}`];
      }
      setOutput(lines);
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }
    setLoading(false);
  };

  const executeDataPosts = async (userId: string) => {
    setLoading(true);
    setOutput([]);
    setScreen("result");
    try {
      const paths = await allDatasetPaths();
      const posts: Post[] = await getPostsByUser(userId, paths);
      if (posts.length === 0) {
        setOutput([`No posts found for "${userId}".`]);
      } else {
        setOutput([
          `Found ${posts.length} post(s) for "${userId}":`,
          "",
          ...posts.map((p) => `[${p.created_at}] ${p.text.slice(0, 100)}${p.text.length > 100 ? "…" : ""}`),
        ]);
      }
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }
    setLoading(false);
  };

  const executeDataMetadata = async (userId: string) => {
    setLoading(true);
    setOutput([]);
    setScreen("result");
    try {
      const paths = await allDatasetPaths();
      const user: User | undefined = await getUserMetadata(userId, paths);
      if (!user) {
        setOutput([`User "${userId}" not found.`]);
      } else {
        setOutput([
          `User metadata for "${userId}":`, "",
          `  ID:          ${user.id}`,
          `  Username:    ${user.username}`,
          `  Name:        ${user.name}`,
          `  Description: ${user.description || "(none)"}`,
          `  Location:    ${user.location || "(none)"}`,
          `  Tweet count: ${user.tweet_count}`,
          `  Z-score:     ${user.z_score.toFixed(4)}`,
        ]);
      }
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }
    setLoading(false);
  };

  const openResumeScreen = () => {
    setResumeLoading(true);
    setScreen("resume-select");
    setResumeCursor(0);
    listIncompleteCaches().then((c) => {
      setResumeCaches(c);
      setResumeLoading(false);
    });
  };

  // ── Detector execution ──

  const runDetectorResult = async (
    fn: any,
    selectedIds: number[],
    model: string,
    modelDisplay: string,
    isRec: boolean,
    isInv: boolean,
    cacheFile?: string,
  ) => {
    try {
      let result: any;
      if (isRec || isInv) {
        const rounds = isInv ? INV_TOTAL_ROUNDS : TOTAL_ROUNDS;
        setIsInvRecursive(isInv);
        setScreen("recursive-detecting");
        setRecursiveProgress({ currentRound: 0, totalRounds: rounds, roundDone: 0, roundTotal: 0, roundLabel: "", completedRounds: [] });
        result = await fn(selectedIds, model, undefined, 50, cacheFile, (p: RecursiveProgress) => setRecursiveProgress(p));
      } else {
        setScreen("detecting");
        setProgress({ done: 0, total: 0 });
        result = await fn(selectedIds, model, (done: number, total: number) => setProgress({ done, total }), undefined, cacheFile);
      }

      const lines: string[] = [
        `Model: ${modelDisplay}`,
        `Datasets: ${selectedIds.join(", ")}`,
        `Users: ${result.totalUsers}`,
        `Bots detected: ${result.botsDetected}`,
        `Humans detected: ${result.humansDetected}`,
        `Run file: ${result.runFile}`,
        "", "--- Accuracy ---", "",
      ];
      const accuracyLines = await analyzeRunFile(result.runFile);
      // Only show summary stats in the UI (details are saved to results/)
      const scoreIdx = accuracyLines.findIndex((l) => l.startsWith("SCORE"));
      const summaryLines = scoreIdx >= 0 ? accuracyLines.slice(0, scoreIdx + 1) : accuracyLines;
      const savedLine = accuracyLines.find((l) => l.startsWith("Results saved to:"));
      lines.push(...summaryLines);
      if (savedLine) lines.push("", savedLine);
      setOutput(lines);
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }
    setScreen("result");
    setLoading(false);
  };

  const startDetector = () => {
    const selectedIds = AVAILABLE_DATASETS.filter((_, i) => dsChecked[i]);
    if (selectedIds.length === 0) return;
    const type = selectedDetector!.type;
    setOutput([]);
    runDetectorResult(detectorFn.fn, selectedIds, detectorModel, detectorModelDisplay, type === "recursive" || type === "recursive-v2", type === "inv_recursive");
  };

  const startDetectorResume = (cache: IncompleteCache) => {
    const fn = getDetectorFn(cache.data.detector);
    const isRec = cache.data.detector === "recursive" || cache.data.detector === "recursive-v2";
    const isInv = cache.data.detector === "inv_recursive";
    setDetectorModelDisplay(cache.data.model);
    setOutput([]);
    runDetectorResult(fn, cache.data.datasetIds, cache.data.model, cache.data.model, isRec, isInv, cache.path);
  };

  // ── Input handling ──

  useInput((input, key) => {
    // Global quit
    if (input === "q" && screen === "menu") {
      exit();
      return;
    }

    // Global search trigger
    if (input === "/" && ["menu", "api-tests", "batch-select", "model-select", "dataset-select"].includes(screen)) {
      openSearch();
      return;
    }

    // ── Search screen ──
    if (screen === "search") {
      if (key.escape) {
        setScreen(returnScreen);
        return;
      }
      if (key.upArrow) {
        setSearchCursor((c) => Math.max(0, c - 1));
        return;
      }
      if (key.downArrow) {
        setSearchCursor((c) => Math.min(Math.max(0, searchResults.length - 1), c + 1));
        return;
      }
      if (key.return && searchResults.length > 0) {
        executeSearchItem(searchResults[searchCursor]!);
        return;
      }
      if (key.backspace || key.delete) {
        setSearchBuf((b) => b.slice(0, -1));
        setSearchCursor(0);
        return;
      }
      if (input && !key.ctrl && !key.meta) {
        setSearchBuf((b) => b + input);
        setSearchCursor(0);
      }
      return;
    }

    // ── Menu screen ──
    if (screen === "menu") {
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setCursor((c) => Math.min(MAIN_MENU.length - 1, c + 1));
      if (key.return) {
        const item = MAIN_MENU[cursor]!;
        if (item.action.startsWith("det-")) {
          const type = item.action.replace("det-", "") as DetectorType;
          const det = DETECTORS.find((d) => d.type === type)!;
          setSelectedDetector(det);
          setSelectedBatched(false);

          if (det.hasBatched) {
            setCursor(0);
            setScreen("batch-select");
          } else if (det.hasModelChoice) {
            setCursor(0);
            setScreen("model-select");
          } else {
            // No batch, no model choice (shouldn't happen currently)
            setupAndGoToDatasets(type, false, det.fixedModelDisplay, det.fixedModelDisplay);
          }
        } else if (item.action === "high-scores") {
          executeHighScores();
        } else if (item.action === "regen-results") {
          executeRegenResults();
        } else if (item.action === "data-posts") {
          showInput("Enter user ID or username", (val) => executeDataPosts(val));
        } else if (item.action === "data-metadata") {
          showInput("Enter user ID or username", (val) => executeDataMetadata(val));
        } else if (item.action === "api-tests") {
          setCursor(0);
          setScreen("api-tests");
        } else if (item.action === "resume") {
          openResumeScreen();
        } else if (item.action === "clear-cache") {
          executeClearCache();
        }
      }
      return;
    }

    // ── API Tests screen ──
    if (screen === "api-tests") {
      if (key.escape) { setCursor(0); setScreen("menu"); return; }
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setCursor((c) => Math.min(API_MENU.length - 1, c + 1));
      if (key.return) {
        const item = API_MENU[cursor]!;
        if (item.action === "data-posts") {
          showInput("Enter user ID or username", (val) => executeDataPosts(val));
        } else if (item.action === "data-metadata") {
          showInput("Enter user ID or username", (val) => executeDataMetadata(val));
        } else {
          executeApiTest(item.action);
        }
      }
      return;
    }

    // ── Batch select screen ──
    if (screen === "batch-select") {
      if (key.escape) { setCursor(0); setScreen("menu"); return; }
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setCursor((c) => Math.min(BATCH_MENU.length - 1, c + 1));
      if (key.return) {
        const batched = cursor === 1;
        setSelectedBatched(batched);

        if (selectedDetector!.hasModelChoice) {
          setCursor(0);
          setScreen("model-select");
        } else {
          // PRG — no model choice, go to datasets
          setupAndGoToDatasets(
            selectedDetector!.type,
            batched,
            selectedDetector!.fixedModelDisplay,
            selectedDetector!.fixedModelDisplay,
          );
        }
      }
      return;
    }

    // ── Model select screen ──
    if (screen === "model-select") {
      if (key.escape) {
        if (selectedDetector!.hasBatched) {
          setCursor(0);
          setScreen("batch-select");
        } else {
          setCursor(0);
          setScreen("menu");
        }
        return;
      }
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setCursor((c) => Math.min(MODEL_CHOICES.length - 1, c + 1));
      if (key.return) {
        const model = MODEL_CHOICES[cursor]!;
        setupAndGoToDatasets(selectedDetector!.type, selectedBatched, model.value, model.name);
      }
      return;
    }

    // ── Dataset select screen ──
    if (screen === "dataset-select") {
      if (key.escape) {
        if (selectedDetector?.hasModelChoice) {
          setCursor(0);
          setScreen("model-select");
        } else if (selectedDetector?.hasBatched) {
          setCursor(0);
          setScreen("batch-select");
        } else {
          setCursor(0);
          setScreen("menu");
        }
        return;
      }
      if (key.upArrow) setDsCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setDsCursor((c) => Math.min(AVAILABLE_DATASETS.length - 1, c + 1));
      if (input === " ") {
        setDsChecked((prev) => {
          const next = [...prev];
          next[dsCursor] = !next[dsCursor]!;
          return next;
        });
      }
      if (key.return && dsChecked.some(Boolean)) {
        startDetector();
      }
      return;
    }

    // ── Resume select ──
    if (screen === "resume-select") {
      if (key.escape) { setCursor(0); setScreen("menu"); return; }
      if (key.upArrow) setResumeCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setResumeCursor((c) => Math.min(Math.max(0, resumeCaches.length - 1), c + 1));
      if (key.return && resumeCaches.length > 0 && resumeCaches[resumeCursor]) {
        startDetectorResume(resumeCaches[resumeCursor]!);
      }
      return;
    }

    // ── Input screen ──
    if (screen === "input") {
      if (key.escape) { setCursor(0); setScreen("menu"); return; }
      if (key.return && inputBuf.trim()) {
        inputCallback.fn(inputBuf.trim());
        return;
      }
      if (key.backspace || key.delete) {
        setInputBuf((b) => b.slice(0, -1));
        return;
      }
      if (input && !key.ctrl && !key.meta) {
        setInputBuf((b) => b + input);
      }
      return;
    }

    // ── Result screen ──
    if (screen === "result") {
      if (key.return || key.escape) { setCursor(0); setScreen("menu"); }
    }
  });

  // ── Render ──────────────────────────────────────────────────────────

  // Search screen
  if (screen === "search") {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">Search</Text>
        <Box marginTop={1}>
          <Text color="yellow">/</Text>
          <Text> {searchBuf}</Text>
          <Text dimColor>▌</Text>
        </Box>
        <Box flexDirection="column" marginTop={1}>
          {searchResults.length === 0 ? (
            <Text dimColor>No matches</Text>
          ) : (
            searchResults.slice(0, 15).map((item, i) => {
              const selected = i === searchCursor;
              return (
                <Box key={`search-${i}`} gap={1}>
                  <Text>{selected ? "▸" : " "}</Text>
                  <Text bold={selected}>{item.name}</Text>
                </Box>
              );
            })
          )}
          {searchResults.length > 15 && (
            <Text dimColor>  … {searchResults.length - 15} more</Text>
          )}
        </Box>
        <Box marginTop={1}>
          <Text dimColor>⏎ select  esc back</Text>
        </Box>
      </Box>
    );
  }

  // Resume select
  if (screen === "resume-select") {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">Resume run</Text>
        <Text dimColor>↑/↓ navigate  ⏎ resume  esc back</Text>
        <Box flexDirection="column" marginTop={1}>
          {resumeLoading ? (
            <Text dimColor>Loading…</Text>
          ) : resumeCaches.length === 0 ? (
            <Text dimColor>No incomplete runs to resume.</Text>
          ) : (
            resumeCaches.map((c, i) => {
              const selected = i === resumeCursor;
              const done = Object.keys(c.data.results).length;
              const pct = c.data.totalUsers > 0 ? Math.round((done / c.data.totalUsers) * 100) : 0;
              return (
                <Box key={c.path} gap={1}>
                  <Text>{selected ? "▸" : " "}</Text>
                  <Text bold={selected}>
                    {c.data.detector} — {c.data.datasetIds.join(",")} — {done}/{c.data.totalUsers} ({pct}%)
                  </Text>
                </Box>
              );
            })
          )}
        </Box>
      </Box>
    );
  }

  // Dataset select
  if (screen === "dataset-select") {
    const anyChecked = dsChecked.some(Boolean);
    const batchLabel = selectedBatched ? " Batched" : "";
    const title = `${selectedDetector?.name}${batchLabel} — ${detectorModelDisplay}`;
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">Select Datasets</Text>
        <Text dimColor>{title}</Text>
        <Text dimColor>↑/↓ navigate  space toggle  ⏎ run  esc back  / search</Text>
        <Box flexDirection="column" marginTop={1}>
          {AVAILABLE_DATASETS.map((id, i) => {
            const selected = i === dsCursor;
            const checked = dsChecked[i];
            return (
              <Box key={`ds-${id}`} gap={1}>
                <Text>{selected ? "▸" : " "}</Text>
                <Text color={checked ? "green" : "gray"}>{checked ? "[x]" : "[ ]"}</Text>
                <Text bold={selected}>Dataset {id}</Text>
              </Box>
            );
          })}
        </Box>
        {anyChecked && (
          <Box marginTop={1}>
            <Text dimColor>Press ⏎ to run</Text>
          </Box>
        )}
      </Box>
    );
  }

  // Detecting (non-recursive)
  if (screen === "detecting") {
    const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
    const barWidth = 30;
    const filled = Math.round((pct / 100) * barWidth);
    const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">Running Detector</Text>
        <Text dimColor>Model: {detectorModelDisplay}</Text>
        <Box marginTop={1} gap={1}>
          <Text color="yellow">{bar}</Text>
          <Text>{progress.done}/{progress.total} users ({pct}%)</Text>
        </Box>
      </Box>
    );
  }

  // Recursive detecting
  if (screen === "recursive-detecting") {
    const barWidth = 30;
    const rp = recursiveProgress;
    const depth = isInvRecursive ? INV_RECURSION_DEPTH : RECURSION_DEPTH;
    const rounds = isInvRecursive ? INV_TOTAL_ROUNDS : TOTAL_ROUNDS;

    const remaining = rp.completedRounds.length > 0
      ? (isInvRecursive
          ? rp.completedRounds[rp.completedRounds.length - 1]!.humansFiltered
          : rp.completedRounds[rp.completedRounds.length - 1]!.botsOut)
      : rp.roundTotal;

    const roundName = (round: number) =>
      round < depth ? `Round ${round + 1}` : "Final batch";
    const roundDesc = (round: number) =>
      round < depth ? "biased" : "baseline";

    const detectorTitle = isInvRecursive ? "Inv Recursive Detector" : "Recursive Detector";
    const poolLabel = isInvRecursive ? "Remaining in human pool" : "Remaining in bot pool";

    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">{detectorTitle}</Text>
        <Text dimColor>Model: {detectorModelDisplay}  Depth: {depth} + final batch</Text>

        <Box flexDirection="column" marginTop={1}>
          {Array.from({ length: rounds }, (_, round) => {
            const completed = rp.completedRounds.find((r) => r.round === round);

            if (completed) {
              const bar = "█".repeat(barWidth);
              const summary = isInvRecursive
                ? `${completed.botsOut} bots filtered, ${completed.humansFiltered} humans remain`
                : `${completed.botsOut} bots, ${completed.humansFiltered} humans filtered`;
              return (
                <Box key={`round-${round}`} flexDirection="column">
                  <Text bold>{roundName(round)} <Text dimColor>({roundDesc(round)})</Text> — {completed.poolSize} users</Text>
                  <Box gap={1}>
                    <Text color="green">{bar}</Text>
                    <Text color="green">{completed.poolSize}/{completed.poolSize} ✓</Text>
                  </Box>
                  <Text dimColor>  → {summary}</Text>
                  <Text> </Text>
                </Box>
              );
            }

            if (round === rp.currentRound && rp.roundTotal > 0) {
              const pct = Math.round((rp.roundDone / rp.roundTotal) * 100);
              const filled = Math.round((pct / 100) * barWidth);
              const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);
              return (
                <Box key={`round-${round}`} flexDirection="column">
                  <Text bold>{roundName(round)} <Text dimColor>({roundDesc(round)})</Text> — {rp.roundTotal} users</Text>
                  <Box gap={1}>
                    <Text color="yellow">{bar}</Text>
                    <Text>{rp.roundDone}/{rp.roundTotal} ({pct}%)</Text>
                  </Box>
                  <Text> </Text>
                </Box>
              );
            }

            return (
              <Box key={`round-${round}`} flexDirection="column">
                <Text dimColor>{roundName(round)} ({roundDesc(round)}) — waiting…</Text>
                <Text> </Text>
              </Box>
            );
          })}
        </Box>

        <Box marginTop={1}>
          <Text bold color="magenta">{poolLabel}: {remaining}</Text>
        </Box>
      </Box>
    );
  }

  // Input screen
  if (screen === "input") {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">{inputLabel}</Text>
        <Text dimColor>Type a value, then press ⏎  (esc to cancel)</Text>
        <Box marginTop={1}>
          <Text>{">"} </Text>
          <Text>{inputBuf}</Text>
          <Text dimColor>▌</Text>
        </Box>
      </Box>
    );
  }

  // Result screen
  if (screen === "result") {
    return (
      <Box flexDirection="column" padding={1}>
        {loading ? (
          <Text color="yellow">Running…</Text>
        ) : (
          <>
            {output.map((line, i) => (
              <Text key={`line-${i}`}>{line}</Text>
            ))}
            <Box marginTop={1}>
              <Text dimColor>Press ⏎ or esc to go back</Text>
            </Box>
          </>
        )}
      </Box>
    );
  }

  // ── List renderer helper ──
  const renderList = (title: string, subtitle: string, items: ListItem[], hint?: string) => (
    <Box flexDirection="column" padding={1}>
      <Text bold color="cyan">{title}</Text>
      <Text dimColor>{subtitle}</Text>
      <Box flexDirection="column" marginTop={1}>
        {items.map((item, i) => {
          const selected = i === cursor;
          return (
            <Box key={item.action} gap={1}>
              <Text>{selected ? "▸" : " "}</Text>
              <Text bold={selected}>{item.name}</Text>
              {item.dimDetail && <Text dimColor> {item.dimDetail}</Text>}
            </Box>
          );
        })}
      </Box>
      {hint && (
        <Box marginTop={1}>
          <Text dimColor>{hint}</Text>
        </Box>
      )}
    </Box>
  );

  // API Tests menu
  if (screen === "api-tests") {
    return renderList("API Tests", "↑/↓ navigate  ⏎ select  esc back  / search", API_MENU);
  }

  // Batch select
  if (screen === "batch-select") {
    return renderList(
      `${selectedDetector?.name} — Mode`,
      "↑/↓ navigate  ⏎ select  esc back  / search",
      BATCH_MENU,
    );
  }

  // Model select
  if (screen === "model-select") {
    const batchLabel = selectedBatched ? " Batched" : "";
    return renderList(
      `${selectedDetector?.name}${batchLabel} — Model`,
      "↑/↓ navigate  ⏎ select  esc back  / search",
      MODEL_CHOICES.map((m) => ({ name: m.name, action: m.value })),
    );
  }

  // Main menu (default)
  return renderList("Bot-or-Not", "↑/↓ navigate  ⏎ select  / search  q quit", MAIN_MENU);
}

render(<App />);
