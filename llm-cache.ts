/**
 * Local KV cache for LLM responses. Key = hash(model + messages), value = response text.
 * Used so identical (model, prompt) requests can return the same result when cache is enabled.
 */

import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";

const DB_PATH = "llm-cache.db";

let db: Database | null = null;

function getDb(): Database {
  if (!db) {
    db = new Database(DB_PATH);
    db.run(
      "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)",
    );
  }
  return db;
}

export type CacheMessage = { role: string; content: string };

/** Build a deterministic cache key from model and messages. */
export function cacheKey(
  model: string,
  messages: CacheMessage[],
): string {
  const payload = JSON.stringify({ model, messages });
  return createHash("sha256").update(payload).digest("hex");
}

/** Get cached response for key, or null if miss. */
export function getCache(key: string): string | null {
  const row = getDb().query<{ response: string }, [string]>(
    "SELECT response FROM cache WHERE key = ?",
  ).get(key);
  return row?.response ?? null;
}

/** Store response for key. */
export function setCache(key: string, value: string): void {
  getDb().run("INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)", [
    key,
    value,
  ]);
}

/** Remove all entries from the cache. */
export function clearCache(): void {
  getDb().run("DELETE FROM cache");
}
