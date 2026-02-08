/**
 * AI SDK tools — callable by the LLM via Vercel AI SDK's tool interface.
 */

import { tool } from "ai";
import { z } from "zod";
import { getPostsByUser, getUserMetadata, type Post, type User } from "./data.ts";
import { Glob } from "bun";

/** Discover all dataset JSON files in the datasets/ directory. */
async function allDatasetPaths(): Promise<string[]> {
  const glob = new Glob("datasets/dataset.posts&users.*.json");
  const paths: string[] = [];
  for await (const path of glob.scan(".")) {
    paths.push(path);
  }
  return paths.sort();
}

/** Tool: list all posts for a given user ID or username across all (or specified) datasets. */
export const listUserPosts = tool({
  description:
    "List all posts/tweets for a given user ID or username. " +
    "Optionally specify which dataset IDs to search (e.g. [30, 31]). " +
    "If none are specified, all available datasets are searched.",
  inputSchema: z.object({
    user: z.string().describe("The user ID or username to look up posts for."),
    datasetIds: z
      .array(z.number())
      .optional()
      .describe("Optional list of dataset IDs to search (e.g. [30, 31, 32, 33]). Defaults to all."),
  }),
  execute: async ({ user, datasetIds }): Promise<Post[]> => {
    let paths: string[];
    if (datasetIds && datasetIds.length > 0) {
      paths = datasetIds.map((id) => `datasets/dataset.posts&users.${id}.json`);
    } else {
      paths = await allDatasetPaths();
    }
    return getPostsByUser(user, paths);
  },
});

/** Tool: get user metadata for a given user ID or username. */
export const getUserInfo = tool({
  description:
    "Get metadata (username, name, description, location, tweet_count, z_score) " +
    "for a given user ID or username. " +
    "Optionally specify which dataset IDs to search.",
  inputSchema: z.object({
    user: z.string().describe("The user ID or username to look up."),
    datasetIds: z
      .array(z.number())
      .optional()
      .describe("Optional list of dataset IDs to search. Defaults to all."),
  }),
  execute: async ({ user, datasetIds }): Promise<User | null> => {
    let paths: string[];
    if (datasetIds && datasetIds.length > 0) {
      paths = datasetIds.map((id) => `datasets/dataset.posts&users.${id}.json`);
    } else {
      paths = await allDatasetPaths();
    }
    return (await getUserMetadata(user, paths)) ?? null;
  },
});
