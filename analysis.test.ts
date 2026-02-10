import { expect, test } from "bun:test";
import { mkdtemp, rm, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { analyzeRunFile } from "./analysis.ts";

test("analyzeRunFile returns a report for the sample run", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "bot-or-not-"));
  const runPath = resolve(tempDir, `test-run-${Date.now()}.txt`);
  const stem = basename(runPath).replace(/\.[^.]+$/, "");
  const resultPath = resolve(import.meta.dir, "results", `${stem}.results.txt`);

  try {
    await Bun.write(
      runPath,
      "Datasets: 30\n00000000-0000-0000-0000-000000000000\n",
    );

    const lines = await analyzeRunFile(runPath);
    expect(lines.some((line) => line === "RUN ACCURACY REPORT")).toBe(true);
    expect(lines.some((line) => line === "Datasets:    [30]")).toBe(true);
    expect(lines.some((line) => line.startsWith("Results saved to: results/"))).toBe(
      true,
    );
  } finally {
    await unlink(runPath).catch(() => {});
    await unlink(resultPath).catch(() => {});
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
});
