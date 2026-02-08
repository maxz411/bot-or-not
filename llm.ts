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

// ── Clients ──────────────────────────────────────────────────────────

// Direct providers
const openaiDirect = new OpenAI({ apiKey: Bun.env.OPENAI_API_KEY });
const anthropicDirect = new Anthropic({ apiKey: Bun.env.ANTHROPIC_API_KEY });

const openaiProvider = createOpenAI({ apiKey: Bun.env.OPENAI_API_KEY });
const anthropicProvider = createAnthropic({ apiKey: Bun.env.ANTHROPIC_API_KEY });

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

/** Retry a function on 429 rate-limit errors with exponential backoff. */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (e: any) {
      const status = e?.status ?? e?.statusCode ?? e?.error?.status;
      if (status === 429 && attempt < MAX_RETRIES - 1) {
        // Parse retry-after hint from error message if available
        const match = String(e.message).match(/try again in (\d+)/i);
        const waitMs = match ? Math.max(parseInt(match[1]!, 10), 100) : 1000 * (attempt + 1);
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
  return openrouterProvider(model) as LanguageModelV3;
}

// ── OpenAI SDK wrappers ──────────────────────────────────────────────

/** Simple chat completion — returns the assistant message string. */
export async function chat(
  prompt: string,
  model = "openai/gpt-4.1-nano",
): Promise<string> {
  if (model.startsWith("anthropic/") || model.startsWith("claude-")) {
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
  }
  const { client, model: m, isDirect } = routeOpenAI(model);
  return withRetry(async () => {
    const res = await client.chat.completions.create({
      model: m,
      messages: [{ role: "user", content: prompt }],
      ...(isDirect ? { service_tier: "flex" as const } : {}),
    });
    return res.choices[0]?.message?.content ?? "";
  });
}

/** Chat with a system message + user message. */
export async function chatWithSystem(
  system: string,
  prompt: string,
  model = "openai/gpt-4.1-nano",
): Promise<string> {
  if (model.startsWith("anthropic/") || model.startsWith("claude-")) {
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
  }
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
    return res.choices[0]?.message?.content ?? "";
  });
}

/** Chat with full message history. */
export async function chatMessages(
  messages: OpenAI.ChatCompletionMessageParam[],
  model = "openai/gpt-4.1-nano",
): Promise<string> {
  if (model.startsWith("anthropic/") || model.startsWith("claude-")) {
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
  }
  const { client, model: m, isDirect } = routeOpenAI(model);
  return withRetry(async () => {
    const res = await client.chat.completions.create({
      model: m,
      messages,
      ...(isDirect ? { service_tier: "flex" as const } : {}),
    });
    return res.choices[0]?.message?.content ?? "";
  });
}

// ── Vercel AI SDK wrappers ───────────────────────────────────────────

/** Generate text (non-streaming) via Vercel AI SDK. */
export async function generate(
  prompt: string,
  model = "openai/gpt-4.1-nano",
): Promise<string> {
  return withRetry(async () => {
    const { text } = await generateText({
      model: routeVercel(model),
      prompt,
    });
    return text;
  });
}

/** Generate with a system + user message via Vercel AI SDK. */
export async function generateWithSystem(
  system: string,
  prompt: string,
  model = "openai/gpt-4.1-nano",
): Promise<string> {
  return withRetry(async () => {
    const { text } = await generateText({
      model: routeVercel(model),
      system,
      prompt,
    });
    return text;
  });
}

/** Generate with full message history via Vercel AI SDK. */
export async function generateFromMessages(
  messages: ModelMessage[],
  model = "openai/gpt-4.1-nano",
): Promise<string> {
  return withRetry(async () => {
    const { text } = await generateText({
      model: routeVercel(model),
      messages,
    });
    return text;
  });
}

/** Stream text via Vercel AI SDK — returns an async iterable of chunks. */
export async function stream(
  prompt: string,
  model = "openai/gpt-4.1-nano",
) {
  const result = streamText({
    model: routeVercel(model),
    prompt,
  });
  return result;
}
