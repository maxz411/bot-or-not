import { afterEach, expect, test } from "bun:test";
import { unlink } from "node:fs/promises";
import { createCache, listIncompleteCaches, loadCache, writeResult } from "./cache.ts";

const createdCachePaths: string[] = [];

function uniqueDetectorName(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

afterEach(async () => {
  await Promise.all(
    createdCachePaths.splice(0).map(async (path) => {
      try {
        await unlink(path);
      } catch {
        // ignore cleanup failures
      }
    }),
  );
});

test("createCache/writeResult/loadCache roundtrip", async () => {
  const path = await createCache(uniqueDetectorName("cache-roundtrip"), "test-model", [30], 2);
  createdCachePaths.push(path);

  const initial = await loadCache(path);
  expect(initial.model).toBe("test-model");
  expect(initial.totalUsers).toBe(2);
  expect(Object.keys(initial.results)).toHaveLength(0);

  initial.results["user-1"] = { isBot: true };
  await writeResult(path, initial);

  const updated = await loadCache(path);
  expect(updated.results["user-1"]?.isBot).toBe(true);
});

test("listIncompleteCaches includes incomplete cache and excludes completed cache", async () => {
  const path = await createCache(uniqueDetectorName("cache-incomplete"), "test-model", [30], 1);
  createdCachePaths.push(path);

  const before = await listIncompleteCaches();
  expect(before.some((entry) => entry.path === path)).toBe(true);

  const data = await loadCache(path);
  data.results["user-1"] = { isBot: false };
  await writeResult(path, data);

  const after = await listIncompleteCaches();
  expect(after.some((entry) => entry.path === path)).toBe(false);
});
