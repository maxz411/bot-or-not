/**
 * Incremental cache for resumable detector runs.
 * Results are written to JSON as they arrive; runs can be resumed by loading the cache.
 */

import { mkdirSync } from "node:fs";

export type CacheData = {
  detector: string;
  model: string;
  datasetIds: number[];
  startedAt: string;
  totalUsers: number;
  results: Record<string, { isBot: boolean }>;
};

const CACHE_DIR = "cache";

function ensureCacheDir(): void {
  try {
    mkdirSync(CACHE_DIR, { recursive: true });
  } catch {
    // ignore
  }
}

/** Create a new cache file and return its path. */
export async function createCache(
  detector: string,
  model: string,
  datasetIds: number[],
  totalUsers: number,
): Promise<string> {
  ensureCacheDir();
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const path = `${CACHE_DIR}/${detector}-${timestamp}.json`;
  const data: CacheData = {
    detector,
    model,
    datasetIds,
    startedAt: new Date().toISOString(),
    totalUsers,
    results: {},
  };
  await Bun.write(path, JSON.stringify(data, null, 0) + "\n");
  return path;
}

/** Load and parse a cache file. */
export async function loadCache(path: string): Promise<CacheData> {
  const file = Bun.file(path);
  const text = await file.text();
  return JSON.parse(text) as CacheData;
}

/** Overwrite the cache file with current state. Silently skips if the file is locked. */
export async function writeResult(path: string, data: CacheData): Promise<void> {
  try {
    await Bun.write(path, JSON.stringify(data, null, 0) + "\n");
  } catch {
    // skip — file may be locked by another process
  }
}

export type IncompleteCache = { path: string; data: CacheData };

/** List cache files that are incomplete (results.length < totalUsers), sorted by mtime descending. */
export async function listIncompleteCaches(): Promise<IncompleteCache[]> {
  ensureCacheDir();
  const glob = new Bun.Glob("*.json");
  const entries: { path: string; mtime: number }[] = [];
  for await (const f of glob.scan(CACHE_DIR)) {
    const full = `${CACHE_DIR}/${f}`;
    const stat = await Bun.file(full).stat().catch(() => null);
    if (stat) entries.push({ path: full, mtime: stat.mtime.getTime() });
  }
  entries.sort((a, b) => b.mtime - a.mtime);

  const out: IncompleteCache[] = [];
  for (const { path } of entries) {
    try {
      const data = await loadCache(path);
      const count = Object.keys(data.results).length;
      if (count < data.totalUsers) {
        out.push({ path, data });
      }
    } catch {
      // skip invalid or unreadable cache files
    }
  }
  return out;
}
