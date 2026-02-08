import React, { useState, useEffect } from "react";
import { render, Box, Text, useInput, useApp } from "ink";
import { chat, generate, stream } from "./llm.ts";
import { getPostsByUser, getUserMetadata, type Post, type User } from "./data.ts";
import { runDetector, MODEL_OPENAI, MODEL_ANTHROPIC } from "./detector.ts";
import { Glob } from "bun";

// ── Helpers ──────────────────────────────────────────────────────────

async function allDatasetPaths(): Promise<string[]> {
  const glob = new Glob("datasets/dataset.posts&users.*.json");
  const paths: string[] = [];
  for await (const path of glob.scan(".")) paths.push(path);
  return paths.sort();
}

// ── Menu items ───────────────────────────────────────────────────────

type MenuItem = { name: string; needsInput?: boolean; action?: string };

const MENU: MenuItem[] = [
  { name: `Baseline OpenAI (${MODEL_OPENAI})`, action: "detector-openai" },
  { name: `Baseline Anthropic (${MODEL_ANTHROPIC})`, action: "detector-anthropic" },
  { name: "API: OpenAI SDK — chat()" },
  { name: "API: Vercel AI  — generate()" },
  { name: "API: Vercel AI  — stream()" },
  { name: "Data: Get user posts", needsInput: true },
  { name: "Data: Get user metadata", needsInput: true },
];

const AVAILABLE_DATASETS = [30, 31, 32, 33];

// ── Components ───────────────────────────────────────────────────────

type Screen = "menu" | "input" | "result" | "dataset-select" | "detecting";

function App() {
  const { exit } = useApp();
  const [screen, setScreen] = useState<Screen>("menu");
  const [cursor, setCursor] = useState(0);
  const [loading, setLoading] = useState(false);
  const [inputBuf, setInputBuf] = useState("");
  const [pendingAction, setPendingAction] = useState<number | null>(null);
  const [output, setOutput] = useState<string[]>([]);

  // Dataset selection state
  const [dsCursor, setDsCursor] = useState(0);
  const [dsChecked, setDsChecked] = useState<boolean[]>(AVAILABLE_DATASETS.map(() => false));

  // Detector state
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [detectorModel, setDetectorModel] = useState(MODEL_OPENAI);

  // ── Run an action (possibly with user input) ──

  const execute = async (idx: number, userInput?: string) => {
    setLoading(true);
    setOutput([]);
    setScreen("result");

    try {
      let lines: string[] = [];

      if (idx === 2) {
        const res = await chat("Say hi in 5 words.");
        lines = [`Response: ${res}`];
      } else if (idx === 3) {
        const res = await generate("Say hi in 5 words.");
        lines = [`Response: ${res}`];
      } else if (idx === 4) {
        const result = await stream("Say hi in 5 words.");
        let text = "";
        for await (const chunk of result.textStream) text += chunk;
        lines = [`Response: ${text}`];
      } else if (idx === 5 && userInput) {
        const paths = await allDatasetPaths();
        const posts: Post[] = await getPostsByUser(userInput, paths);
        if (posts.length === 0) {
          lines = [`No posts found for "${userInput}".`];
        } else {
          lines = [
            `Found ${posts.length} post(s) for "${userInput}":`,
            "",
            ...posts.map(
              (p) => `[${p.created_at}] ${p.text.slice(0, 100)}${p.text.length > 100 ? "…" : ""}`,
            ),
          ];
        }
      } else if (idx === 6 && userInput) {
        const paths = await allDatasetPaths();
        const user: User | undefined = await getUserMetadata(userInput, paths);
        if (!user) {
          lines = [`User "${userInput}" not found.`];
        } else {
          lines = [
            `User metadata for "${userInput}":`,
            "",
            `  ID:          ${user.id}`,
            `  Username:    ${user.username}`,
            `  Name:        ${user.name}`,
            `  Description: ${user.description || "(none)"}`,
            `  Location:    ${user.location || "(none)"}`,
            `  Tweet count: ${user.tweet_count}`,
            `  Z-score:     ${user.z_score.toFixed(4)}`,
          ];
        }
      }

      setOutput(lines);
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }

    setLoading(false);
  };

  // ── Run detector flow ──

  const startDetector = async () => {
    const selectedIds = AVAILABLE_DATASETS.filter((_, i) => dsChecked[i]);
    if (selectedIds.length === 0) return;

    setScreen("detecting");
    setProgress({ done: 0, total: 0 });
    setOutput([]);

    try {
      const result = await runDetector(selectedIds, detectorModel, (done, total) => {
        setProgress({ done, total });
      });

      // Run the python accuracy checker
      const lines: string[] = [
        `Model: ${detectorModel}`,
        `Datasets: ${selectedIds.join(", ")}`,
        `Users: ${result.totalUsers}`,
        `Bots detected: ${result.botsDetected}`,
        `Humans detected: ${result.humansDetected}`,
        `Run file: ${result.runFile}`,
        "",
        "--- Accuracy (python main.py) ---",
        "",
      ];

      const proc = Bun.spawn(["python3", "main.py", result.runFile], {
        stdout: "pipe",
        stderr: "pipe",
      });
      const stdout = await new Response(proc.stdout).text();
      const stderr = await new Response(proc.stderr).text();
      await proc.exited;

      if (stdout.trim()) {
        lines.push(...stdout.trim().split("\n"));
      }
      if (stderr.trim()) {
        lines.push("", "Errors:", ...stderr.trim().split("\n"));
      }

      setOutput(lines);
    } catch (e: any) {
      setOutput([`Error: ${e.message ?? String(e)}`]);
    }

    setScreen("result");
    setLoading(false);
  };

  // ── Input handling ──

  useInput((input, key) => {
    if (input === "q" && screen === "menu") {
      exit();
      return;
    }

    if (screen === "menu") {
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      if (key.downArrow) setCursor((c) => Math.min(MENU.length - 1, c + 1));
      if (key.return) {
        const item = MENU[cursor]!;
        if (item.action === "detector-openai") {
          setDetectorModel(MODEL_OPENAI);
          setDsCursor(0);
          setDsChecked(AVAILABLE_DATASETS.map(() => false));
          setScreen("dataset-select");
        } else if (item.action === "detector-anthropic") {
          setDetectorModel(MODEL_ANTHROPIC);
          setDsCursor(0);
          setDsChecked(AVAILABLE_DATASETS.map(() => false));
          setScreen("dataset-select");
        } else if (item.needsInput) {
          setPendingAction(cursor);
          setInputBuf("");
          setScreen("input");
        } else {
          execute(cursor);
        }
      }
      return;
    }

    if (screen === "dataset-select") {
      if (key.escape) {
        setScreen("menu");
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
      if (key.return) {
        const anyChecked = dsChecked.some(Boolean);
        if (anyChecked) startDetector();
      }
      return;
    }

    if (screen === "input") {
      if (key.escape) {
        setScreen("menu");
        return;
      }
      if (key.return && inputBuf.trim()) {
        execute(pendingAction!, inputBuf.trim());
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

    if (screen === "result") {
      if (key.return || key.escape) {
        setScreen("menu");
      }
    }
  });

  // ── Render ──

  if (screen === "dataset-select") {
    const anyChecked = dsChecked.some(Boolean);
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">Run Baseline Detector</Text>
        <Text dimColor>↑/↓ navigate  space toggle  ⏎ run  esc back</Text>
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
            <Text dimColor>Press ⏎ to run detector on selected datasets</Text>
          </Box>
        )}
      </Box>
    );
  }

  if (screen === "detecting") {
    const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
    const barWidth = 30;
    const filled = Math.round((pct / 100) * barWidth);
    const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">Running Baseline Detector</Text>
        <Text dimColor>Model: {detectorModel}</Text>
        <Box marginTop={1} gap={1}>
          <Text color="yellow">{bar}</Text>
          <Text>{progress.done}/{progress.total} users ({pct}%)</Text>
        </Box>
      </Box>
    );
  }

  if (screen === "input") {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">{MENU[pendingAction!]!.name}</Text>
        <Text dimColor>Enter a user ID or username, then press ⏎  (esc to cancel)</Text>
        <Box marginTop={1}>
          <Text>{">"} </Text>
          <Text>{inputBuf}</Text>
          <Text dimColor>▌</Text>
        </Box>
      </Box>
    );
  }

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

  return (
    <Box flexDirection="column" padding={1}>
      <Text bold color="cyan">Bot-or-Not CLI</Text>
      <Text dimColor>↑/↓ navigate  ⏎ select  q quit</Text>
      <Box flexDirection="column" marginTop={1}>
        {MENU.map((item, i) => {
          const selected = i === cursor;
          return (
            <Box key={item.name} gap={1}>
              <Text>{selected ? "▸" : " "}</Text>
              <Text bold={selected}>{item.name}</Text>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

render(<App />);
