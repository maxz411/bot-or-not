/** Centralized model constants — all detectors import from here. */

const OPENAI_FT_MODEL_V5 = Bun.env.OPENAI_FT_MODEL_V5?.trim();

export const MODEL_OPENAI = "openai/gpt-5.2";
export const MODEL_OPENAI_4_1 = "openai/gpt-4.1-2025-04-14";
export const MODEL_OPENAI_FT_ALL = "openai/ft:gpt-4.1-mini-2025-04-14:personal:bot-or-not-all:D91Q96GR";
export const MODEL_OPENAI_FT_TEST = "openai/ft:gpt-4.1-mini-2025-04-14:personal:test:D91ewpzh";
export const MODEL_OPENAI_V5 = OPENAI_FT_MODEL_V5
  ? (OPENAI_FT_MODEL_V5.startsWith("openai/")
    ? OPENAI_FT_MODEL_V5
    : `openai/${OPENAI_FT_MODEL_V5}`)
  : "openai/gpt-4.1-mini";
export const MODEL_ANTHROPIC = "anthropic/claude-haiku-4-5";
export const MODEL_ANTHROPIC_SONNET = "anthropic/claude-sonnet-4-5";
export const MODEL_ANTHROPIC_OPUS = "anthropic/claude-opus-4-6";
export const MODEL_DEEPSEEK = "deepseek/deepseek-v3.2-speciale";
export const MODEL_GEMINI = "google/gemini-3-flash-preview";
export const MODEL_GEMINI_PRO = "google/gemini-3-pro-preview";
export const MODEL_GLM = "z-ai/glm-4.7";
export const MODEL_KIMI = "moonshotai/kimi-k2.5";
export const MODEL_GROK = "x-ai/grok-4.1-fast";
export const MODEL_MISTRAL_LARGE = "mistralai/mistral-large-2512";
