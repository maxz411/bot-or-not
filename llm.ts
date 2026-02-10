/**
 * LLM interface layer — routes to the right provider based on model prefix:
 *   - "openai/..."    → OpenAI directly
 *   - "anthropic/..." or "claude-..." → Anthropic directly
 *   - anything else   → OpenRouter
 */

import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { generateText, streamText, type ModelMessage } from "ai";
import type { LanguageModelV3 } from "@ai-sdk/provider";
import { cacheKey, getCache, setCache, type CacheMessage } from "./llm-cache.ts";

export type LLMOptions = { cache?: boolean };

// ── Clients ──────────────────────────────────────────────────────────

// Direct providers
const openaiDirect = new OpenAI({ apiKey: Bun.env.OPENAI_API_KEY });
const anthropicDirect = new Anthropic({ apiKey: Bun.env.ANTHROPIC_API_KEY });

const openaiProvider = createOpenAI({ apiKey: Bun.env.OPENAI_API_KEY });
const anthropicProvider = createAnthropic({ apiKey: Bun.env.ANTHROPIC_API_KEY });

// Groq (OpenAI-compatible)
const groqClient = new OpenAI({
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: Bun.env.GROQ_API_KEY,
});
const groqProvider = createOpenAI({
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: Bun.env.GROQ_API_KEY,
});

// OpenRouter fallback
const openrouterClient = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: Bun.env.OPENROUTER_API_KEY,
});
const openrouterProvider = createOpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: Bun.env.OPENROUTER_API_KEY,
});

// ── Retry on rate limit ──────────────────────────────────────────────

const MAX_RETRIES = 5;
const MIN_RETRY_DELAY_MS = 1000;
const MAX_FALLBACK_DELAY_MS = 60_000;

let globalRateLimitUntil = 0;

function getStatusCode(err: any): number | undefined {
  return err?.status ?? err?.statusCode ?? err?.error?.status ?? err?.response?.status;
}

function getHeader(err: any, name: string): string | undefined {
  const headers =
    err?.headers ??
    err?.response?.headers ??
    err?.error?.headers ??
    err?.error?.response?.headers;
  if (!headers) return undefined;
  const key = name.toLowerCase();
  if (typeof headers.get === "function") {
    return headers.get(name) ?? headers.get(key) ?? undefined;
  }
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === key) {
      return Array.isArray(v) ? String(v[0]) : String(v);
    }
  }
  return undefined;
}

function parseDurationMs(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const unitRegex = /(\d+(?:\.\d+)?)(ms|s|m|h)/gi;
  let total = 0;
  let matched = false;
  let match: RegExpExecArray | null;
  while ((match = unitRegex.exec(trimmed))) {
    matched = true;
    const amount = Number(match[1]);
    const unit = (match[2] ?? "s").toLowerCase();
    if (!Number.isFinite(amount)) continue;
    if (unit === "ms") total += amount;
    if (unit === "s") total += amount * 1000;
    if (unit === "m") total += amount * 60_000;
    if (unit === "h") total += amount * 3_600_000;
  }
  if (matched) return Math.round(total);
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    return Math.round(Number(trimmed) * 1000);
  }
  return null;
}

function parseRetryAfterMs(value?: string): number | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    return Math.round(Number(trimmed) * 1000);
  }
  const asDate = Date.parse(trimmed);
  if (!Number.isNaN(asDate)) {
    const delta = asDate - Date.now();
    return delta > 0 ? delta : 0;
  }
  return parseDurationMs(trimmed);
}

function parseRetryAfterFromMessage(message: string): number | null {
  const match = message.match(/(?:try again in|retry after)\s+(\d+(?:\.\d+)?)(ms|s|m)?/i);
  if (!match) return null;
  const amount = Number(match[1]);
  const unit = (match[2] ?? "s").toLowerCase();
  if (!Number.isFinite(amount)) return null;
  if (unit === "ms") return Math.round(amount);
  if (unit === "m") return Math.round(amount * 60_000);
  return Math.round(amount * 1000);
}

function addJitter(ms: number): number {
  const jitter = Math.min(ms * 0.2, 1000);
  return Math.round(ms + Math.random() * jitter);
}

function computeRetryDelayMs(err: any, attempt: number): number {
  const retryAfter = parseRetryAfterMs(getHeader(err, "retry-after"));
  const resetAfter = parseDurationMs(getHeader(err, "x-ratelimit-reset-requests") ?? "");
  const messageAfter = parseRetryAfterFromMessage(String(err?.message ?? err?.error?.message ?? ""));
  const hinted = retryAfter ?? resetAfter ?? messageAfter;
  const base =
    hinted ??
    Math.min(
      MAX_FALLBACK_DELAY_MS,
      MIN_RETRY_DELAY_MS * Math.pow(2, attempt),
    );
  return Math.max(MIN_RETRY_DELAY_MS, addJitter(base));
}

async function waitForGlobalRateLimit(): Promise<void> {
  const now = Date.now();
  if (globalRateLimitUntil > now) {
    await Bun.sleep(globalRateLimitUntil - now);
  }
}

function extendGlobalRateLimit(waitMs: number): void {
  const until = Date.now() + waitMs;
  if (until > globalRateLimitUntil) {
    globalRateLimitUntil = until;
  }
}

/** Error class for empty/malformed API responses that should be retried. */
class EmptyResponseError extends Error {
  status = 502;
  constructor() { super("Empty or malformed API response"); }
}

/** Retry a function on 429 rate-limit errors with RPM-friendly backoff. */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    await waitForGlobalRateLimit();
    try {
      return await fn();
    } catch (e: any) {
      const status = getStatusCode(e);
      const retryable = status === 429 || status === 500 || status === 502 || status === 503;
      if (retryable && attempt < MAX_RETRIES - 1) {
        const waitMs = status === 429
          ? computeRetryDelayMs(e, attempt)
          : Math.min(MAX_FALLBACK_DELAY_MS, MIN_RETRY_DELAY_MS * Math.pow(2, attempt));
        if (status === 429) extendGlobalRateLimit(waitMs);
        await Bun.sleep(waitMs);
        continue;
      }
      throw e;
    }
  }
  throw new Error("withRetry: unreachable");
}

// ── Routing helpers ──────────────────────────────────────────────────

/** Strip provider prefix from model string (e.g. "openai/gpt-4.1" → "gpt-4.1"). */
function stripPrefix(model: string): string {
  const slash = model.indexOf("/");
  return slash >= 0 ? model.slice(slash + 1) : model;
}

/** Pick the right OpenAI SDK client + model name based on model string. */
function routeOpenAI(model: string): { client: OpenAI; model: string; isDirect: boolean } {
  if (model.startsWith("openai/")) {
    return { client: openaiDirect, model: stripPrefix(model), isDirect: true };
  }
  if (model.startsWith("groq/")) {
    return { client: groqClient, model: stripPrefix(model), isDirect: false };
  }
  // anthropic models don't go through OpenAI SDK
  return { client: openrouterClient, model, isDirect: false };
}

/** Pick the right Vercel AI SDK provider model based on model string. */
function routeVercel(model: string): LanguageModelV3 {
  if (model.startsWith("openai/")) {
    return openaiProvider(stripPrefix(model)) as LanguageModelV3;
  }
  if (model.startsWith("anthropic/") || model.startsWith("claude-")) {
    const name = model.startsWith("anthropic/") ? stripPrefix(model) : model;
    return anthropicProvider(name) as LanguageModelV3;
  }
  if (model.startsWith("groq/")) {
    return groqProvider(stripPrefix(model)) as LanguageModelV3;
  }
  return openrouterProvider(model) as LanguageModelV3;
}

// ── OpenAI SDK wrappers ──────────────────────────────────────────────

/** Simple chat completion — returns the assistant message string. */
export async function chat(
  prompt: string,
  model = "openai/gpt-4.1-nano",
  options?: LLMOptions,
): Promise<string> {
  const messages: CacheMessage[] = [{ role: "user", content: prompt }];
  const key = cacheKey(model, messages);
  if (options?.cache) {
    const cached = getCache(key);
    if (cached != null) return cached;
  }
  const result = await (model.startsWith("anthropic/") || model.startsWith("claude-")
    ? (async () => {
        const name = model.startsWith("anthropic/") ? stripPrefix(model) : model;
        return withRetry(async () => {
          const res = await anthropicDirect.messages.create({
            model: name,
            max_tokens: 1024,
            messages: [{ role: "user", content: prompt }],
          });
          const block = res.content[0];
          return block?.type === "text" ? block.text : "";
        });
      })()
    : (async () => {
        const { client, model: m, isDirect } = routeOpenAI(model);
        return withRetry(async () => {
          const res = await client.chat.completions.create({
            model: m,
            messages: [{ role: "user", content: prompt }],
            ...(isDirect ? { service_tier: "flex" as const } : {}),
          });
          const text = res?.choices?.[0]?.message?.content;
          if (text == null) throw new EmptyResponseError();
          return text;
        });
      })());
  setCache(key, result);
  return result;
}

/** Chat with a system message + user message. */
export async function chatWithSystem(
  system: string,
  prompt: string,
  model = "openai/gpt-4.1-nano",
  options?: LLMOptions,
): Promise<string> {
  const messages: CacheMessage[] = [
    { role: "system", content: system },
    { role: "user", content: prompt },
  ];
  const key = cacheKey(model, messages);
  if (options?.cache) {
    const cached = getCache(key);
    if (cached != null) return cached;
  }
  const result = await (model.startsWith("anthropic/") || model.startsWith("claude-")
    ? (async () => {
        const name = model.startsWith("anthropic/") ? stripPrefix(model) : model;
        return withRetry(async () => {
          const res = await anthropicDirect.messages.create({
            model: name,
            max_tokens: 1024,
            system,
            messages: [{ role: "user", content: prompt }],
          });
          const block = res.content[0];
          return block?.type === "text" ? block.text : "";
        });
      })()
    : (async () => {
        const { client, model: m, isDirect } = routeOpenAI(model);
        return withRetry(async () => {
          const res = await client.chat.completions.create({
            model: m,
            messages: [
              { role: "system", content: system },
              { role: "user", content: prompt },
            ],
            ...(isDirect ? { service_tier: "flex" as const } : {}),
          });
          const text = res?.choices?.[0]?.message?.content;
          if (text == null) throw new EmptyResponseError();
          return text;
        });
      })());
  setCache(key, result);
  return result;
}

/** Chat with full message history. */
export async function chatMessages(
  messages: OpenAI.ChatCompletionMessageParam[],
  model = "openai/gpt-4.1-nano",
  options?: LLMOptions,
): Promise<string> {
  const cacheMessages: CacheMessage[] = messages.map((m) => ({
    role: m.role,
    content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
  }));
  const key = cacheKey(model, cacheMessages);
  if (options?.cache) {
    const cached = getCache(key);
    if (cached != null) return cached;
  }
  const result = await (model.startsWith("anthropic/") || model.startsWith("claude-")
    ? (async () => {
        const name = model.startsWith("anthropic/") ? stripPrefix(model) : model;
        const anthropicMsgs = messages
          .filter((m): m is { role: "user" | "assistant"; content: string } =>
            m.role === "user" || m.role === "assistant",
          )
          .map((m) => ({ role: m.role, content: String(m.content) }));
        const systemMsg = messages.find((m) => m.role === "system");
        return withRetry(async () => {
          const res = await anthropicDirect.messages.create({
            model: name,
            max_tokens: 1024,
            ...(systemMsg ? { system: String(systemMsg.content) } : {}),
            messages: anthropicMsgs,
          });
          const block = res.content[0];
          return block?.type === "text" ? block.text : "";
        });
      })()
    : (async () => {
        const { client, model: m, isDirect } = routeOpenAI(model);
        return withRetry(async () => {
          const res = await client.chat.completions.create({
            model: m,
            messages,
            ...(isDirect ? { service_tier: "flex" as const } : {}),
          });
          const text = res?.choices?.[0]?.message?.content;
          if (text == null) throw new EmptyResponseError();
          return text;
        });
      })());
  setCache(key, result);
  return result;
}

// ── Vercel AI SDK wrappers ───────────────────────────────────────────

/** Generate text (non-streaming) via Vercel AI SDK. */
export async function generate(
  prompt: string,
  model = "openai/gpt-4.1-nano",
  options?: LLMOptions,
): Promise<string> {
  const messages: CacheMessage[] = [{ role: "user", content: prompt }];
  const key = cacheKey(model, messages);
  if (options?.cache) {
    const cached = getCache(key);
    if (cached != null) return cached;
  }
  const result = await withRetry(async () => {
    const { text } = await generateText({
      model: routeVercel(model),
      prompt,
    });
    return text;
  });
  setCache(key, result);
  return result;
}

/** Generate with a system + user message via Vercel AI SDK. */
export async function generateWithSystem(
  system: string,
  prompt: string,
  model = "openai/gpt-4.1-nano",
  options?: LLMOptions,
): Promise<string> {
  const messages: CacheMessage[] = [
    { role: "system", content: system },
    { role: "user", content: prompt },
  ];
  const key = cacheKey(model, messages);
  if (options?.cache) {
    const cached = getCache(key);
    if (cached != null) return cached;
  }
  const result = await withRetry(async () => {
    const { text } = await generateText({
      model: routeVercel(model),
      system,
      prompt,
    });
    return text;
  });
  setCache(key, result);
  return result;
}

/** Generate with full message history via Vercel AI SDK. */
export async function generateFromMessages(
  messages: ModelMessage[],
  model = "openai/gpt-4.1-nano",
  options?: LLMOptions,
): Promise<string> {
  const cacheMessages: CacheMessage[] = messages.map((m) => ({
    role: m.role,
    content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
  }));
  const key = cacheKey(model, cacheMessages);
  if (options?.cache) {
    const cached = getCache(key);
    if (cached != null) return cached;
  }
  const result = await withRetry(async () => {
    const { text } = await generateText({
      model: routeVercel(model),
      messages,
    });
    return text;
  });
  setCache(key, result);
  return result;
}

/** Stream text via Vercel AI SDK — returns an async iterable of chunks. */
export async function stream(
  prompt: string,
  model = "openai/gpt-4.1-nano",
) {
  return withRetry(async () =>
    streamText({
      model: routeVercel(model),
      prompt,
    }),
  );
}
