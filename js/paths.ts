/**
 * Centralized path constants — resolves all shared directories relative
 * to the js/ folder so paths work regardless of CWD.
 */

import { resolve } from "path";

/** Project root — parent of js/. */
export const PROJECT_ROOT = resolve(import.meta.dir, "..");

/** Shared directories at project root. */
export const DATASETS_DIR = resolve(PROJECT_ROOT, "datasets");
export const RESULTS_DIR = resolve(PROJECT_ROOT, "results");
export const RUNS_DIR = resolve(PROJECT_ROOT, "runs");

/** JS-local directories. */
export const CACHE_DIR = resolve(import.meta.dir, "cache");

/** Helper: absolute path to a dataset file. */
export function datasetPath(id: number): string {
  return resolve(DATASETS_DIR, `dataset.posts&users.${id}.json`);
}

/** Helper: absolute paths for multiple dataset IDs. */
export function datasetPaths(ids: number[]): string[] {
  return ids.map(datasetPath);
}
