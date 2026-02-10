/**
 * Time-based subdetector — analyzes posting timestamps to infer sleep patterns
 * and estimate the user's timezone/location from activity gaps.
 *
 * This is a utility subdetector: it doesn't classify bot/human on its own,
 * but provides structured time analysis for other detectors to consume.
 */

import type { Post } from "../../data.ts";

// ── Output types ─────────────────────────────────────────────────────

export type TimeAnalysis = {
  /** Estimated UTC offset in hours (e.g. -5 for EST, +9 for JST). null if not enough data. */
  estimatedUtcOffset: number | null;
  /** Human-readable region guess based on the estimated offset. */
  estimatedRegion: string | null;
  /** Whether a clear sleep-like gap (≥4h quiet window) was found. */
  hasSleepGap: boolean;
  /** Length of the longest consecutive quiet period in hours. */
  longestGapHours: number;
  /** Average length of detected daily sleep gaps in hours. */
  avgSleepGapHours: number;
  /** Number of distinct ~daily sleep periods found in the data. */
  sleepPeriodsDetected: number;
  /** Posts per hour during estimated awake window. */
  awakePostsPerHour: number;
  /** Posts per hour during estimated sleep window (should be ~0 for humans). */
  sleepPostsPerHour: number;
  /** Ratio of sleep-hour activity to awake-hour activity. 0 = perfect human, 1 = flat/no sleep. */
  sleepWakeRatio: number;
  /** How consistent the sleep gap is across days (0 = erratic, 1 = very consistent). */
  sleepConsistency: number;
  /** Confidence in the overall analysis (0–1). Low when few posts or short time span. */
  confidence: number;
  /** One-line human-readable summary for other detectors to include in prompts. */
  summary: string;
};

// ── Timezone → region mapping ────────────────────────────────────────

const UTC_OFFSET_REGIONS: Record<number, string> = {
  [-12]: "Baker Island",
  [-11]: "American Samoa",
  [-10]: "Hawaii",
  [-9]: "Alaska",
  [-8]: "US Pacific (LA/SF/Seattle)",
  [-7]: "US Mountain (Denver/Phoenix)",
  [-6]: "US Central (Chicago/Dallas)",
  [-5]: "US Eastern (NYC/Miami)",
  [-4]: "Atlantic (Puerto Rico/Eastern Caribbean)",
  [-3]: "Brazil/Argentina",
  [-2]: "Mid-Atlantic",
  [-1]: "Azores/Cape Verde",
  [0]: "UK/Portugal/West Africa",
  [1]: "Central Europe (Paris/Berlin/Madrid)",
  [2]: "Eastern Europe (Athens/Cairo/Johannesburg)",
  [3]: "Moscow/Saudi Arabia/East Africa",
  [4]: "Gulf States (Dubai/Baku)",
  [5]: "Pakistan/Western Central Asia",
  [5.5]: "India/Sri Lanka",
  [6]: "Bangladesh/Central Asia",
  [7]: "Southeast Asia (Bangkok/Jakarta)",
  [8]: "China/Singapore/Western Australia",
  [9]: "Japan/Korea",
  [9.5]: "Central Australia",
  [10]: "Eastern Australia (Sydney/Melbourne)",
  [11]: "Pacific Islands (New Caledonia)",
  [12]: "New Zealand/Fiji",
};

function regionFromOffset(offset: number): string {
  // Try exact match first, then nearest
  if (UTC_OFFSET_REGIONS[offset]) return UTC_OFFSET_REGIONS[offset];
  const nearest = Object.keys(UTC_OFFSET_REGIONS)
    .map(Number)
    .reduce((best, k) => (Math.abs(k - offset) < Math.abs(best - offset) ? k : best));
  return UTC_OFFSET_REGIONS[nearest] ?? "Unknown";
}

// ── Core analysis ────────────────────────────────────────────────────

/** Build a 24-bin histogram of post counts by UTC hour. */
function buildHourHistogram(timestamps: Date[]): number[] {
  const bins = new Array(24).fill(0);
  for (const ts of timestamps) {
    bins[ts.getUTCHours()]++;
  }
  return bins;
}

/**
 * Find the quietest consecutive window of `windowSize` hours in a 24-hour
 * histogram (wrapping around midnight). Returns the starting UTC hour.
 */
function findQuietestWindow(histogram: number[], windowSize: number): { startHour: number; totalPosts: number } {
  let bestStart = 0;
  let bestTotal = Infinity;

  for (let start = 0; start < 24; start++) {
    let total = 0;
    for (let i = 0; i < windowSize; i++) {
      total += histogram[(start + i) % 24]!;
    }
    if (total < bestTotal) {
      bestTotal = total;
      bestStart = start;
    }
  }

  return { startHour: bestStart, totalPosts: bestTotal };
}

/**
 * Estimate UTC offset from the quietest window. Assumes humans sleep roughly
 * 23:00–06:00 local time, so the center of the quiet window ≈ 02:30 local.
 * We solve: quietCenter_UTC - offset = 2.5 → offset = quietCenter_UTC - 2.5
 */
function estimateUtcOffset(quietStartHour: number, windowSize: number): number {
  const quietCenter = (quietStartHour + windowSize / 2) % 24;
  let offset = quietCenter - 2.5; // 2:30 AM local = assumed sleep midpoint
  // Normalize to [-12, 12]
  if (offset > 12) offset -= 24;
  if (offset < -12) offset += 24;
  return Math.round(offset * 2) / 2; // round to nearest half-hour
}

/**
 * Find per-day sleep gaps. For each calendar day (UTC), find the longest
 * stretch of no posts. Returns the gap lengths in hours.
 */
function findDailyGaps(timestamps: Date[]): number[] {
  if (timestamps.length < 2) return [];

  // Group by calendar date (shifted by estimated local offset later — here just use UTC)
  const byDay = new Map<string, number[]>();
  for (const ts of timestamps) {
    const key = ts.toISOString().slice(0, 10);
    if (!byDay.has(key)) byDay.set(key, []);
    byDay.get(key)!.push(ts.getTime());
  }

  const gaps: number[] = [];
  for (const [, times] of byDay) {
    if (times.length < 2) continue;
    times.sort((a, b) => a - b);
    let maxGap = 0;
    for (let i = 1; i < times.length; i++) {
      maxGap = Math.max(maxGap, times[i]! - times[i - 1]!);
    }
    gaps.push(maxGap / (1000 * 60 * 60)); // ms → hours
  }

  return gaps;
}

// ── Public API ───────────────────────────────────────────────────────

/**
 * Analyze a user's posting timestamps and return structured time intelligence.
 * Pass all posts for the user (already fetched). No network calls.
 */
export function analyzePostingTimes(posts: Post[]): TimeAnalysis {
  const timestamps = posts
    .map((p) => new Date(p.created_at))
    .filter((d) => !isNaN(d.getTime()))
    .sort((a, b) => a.getTime() - b.getTime());

  // ── Not enough data ──
  if (timestamps.length < 3) {
    return {
      estimatedUtcOffset: null,
      estimatedRegion: null,
      hasSleepGap: false,
      longestGapHours: 0,
      avgSleepGapHours: 0,
      sleepPeriodsDetected: 0,
      awakePostsPerHour: 0,
      sleepPostsPerHour: 0,
      sleepWakeRatio: 1,
      sleepConsistency: 0,
      confidence: 0,
      summary: "Not enough posts to analyze time patterns.",
    };
  }

  const histogram = buildHourHistogram(timestamps);
  const totalPosts = timestamps.length;

  // Try window sizes from 6 to 8 hours (typical sleep durations)
  const windows = [6, 7, 8].map((size) => ({
    size,
    ...findQuietestWindow(histogram, size),
  }));

  // Pick the window with the lowest post density (posts per hour in window)
  const bestWindow = windows.reduce((best, w) =>
    w.totalPosts / w.size < best.totalPosts / best.size ? w : best,
  );

  const utcOffset = estimateUtcOffset(bestWindow.startHour, bestWindow.size);

  // Sleep vs awake activity
  const sleepHours = new Set<number>();
  for (let i = 0; i < bestWindow.size; i++) {
    sleepHours.add((bestWindow.startHour + i) % 24);
  }
  const awakeHours = 24 - bestWindow.size;

  const sleepPosts = bestWindow.totalPosts;
  const awakePosts = totalPosts - sleepPosts;

  // Time span in hours for rate calculation
  const spanHours = Math.max(
    1,
    (timestamps[timestamps.length - 1]!.getTime() - timestamps[0]!.getTime()) / (1000 * 60 * 60),
  );
  const spanDays = spanHours / 24;

  const awakePostsPerHour = spanDays > 0 ? awakePosts / (awakeHours * spanDays) : 0;
  const sleepPostsPerHour = spanDays > 0 ? sleepPosts / (bestWindow.size * spanDays) : 0;

  // Sleep/wake ratio: 0 if no sleep posts, 1 if perfectly flat
  const sleepWakeRatio =
    awakePostsPerHour > 0 ? Math.min(1, sleepPostsPerHour / awakePostsPerHour) : 1;

  // Per-day gap analysis
  const dailyGaps = findDailyGaps(timestamps);
  const longestGapHours = dailyGaps.length > 0 ? Math.max(...dailyGaps) : 0;
  const avgSleepGapHours =
    dailyGaps.length > 0 ? dailyGaps.reduce((a, b) => a + b, 0) / dailyGaps.length : 0;

  // Count sleep periods: days with a gap ≥ 4 hours
  const sleepPeriodsDetected = dailyGaps.filter((g) => g >= 4).length;

  // Sleep consistency: std dev of daily gap lengths (lower = more consistent)
  let sleepConsistency = 0;
  if (dailyGaps.length >= 2) {
    const mean = avgSleepGapHours;
    const variance = dailyGaps.reduce((sum, g) => sum + (g - mean) ** 2, 0) / dailyGaps.length;
    const stdDev = Math.sqrt(variance);
    // Map std dev to 0–1 score (0h std = 1.0 consistency, 6h+ std = 0.0)
    sleepConsistency = Math.max(0, 1 - stdDev / 6);
  }

  // Has a real sleep gap?
  const hasSleepGap = bestWindow.totalPosts / totalPosts < 0.05 && bestWindow.size >= 6;

  // Confidence based on data quantity and time span
  const postCountFactor = Math.min(1, totalPosts / 20); // need ~20 posts for decent analysis
  const spanFactor = Math.min(1, spanDays / 2); // need ~2 days for day/night pattern
  const confidence = Math.round(postCountFactor * spanFactor * 100) / 100;

  // Region guess
  const estimatedRegion = regionFromOffset(utcOffset);

  // ── Build summary ──
  const parts: string[] = [];

  if (confidence < 0.3) {
    parts.push(`Low confidence (${(confidence * 100).toFixed(0)}%) — limited data.`);
  }

  if (hasSleepGap) {
    parts.push(
      `Clear sleep gap detected (~${bestWindow.size}h quiet window).`,
      `Estimated timezone: UTC${utcOffset >= 0 ? "+" : ""}${utcOffset} (${estimatedRegion}).`,
    );
  } else {
    parts.push(`No clear sleep gap — posts spread across all hours.`);
    if (sleepWakeRatio > 0.5) {
      parts.push(`Activity is nearly flat across day/night (ratio: ${sleepWakeRatio.toFixed(2)}).`);
    }
  }

  if (sleepPeriodsDetected > 0) {
    parts.push(
      `${sleepPeriodsDetected} daily sleep period(s) found, avg gap ${avgSleepGapHours.toFixed(1)}h.`,
    );
  }

  parts.push(`Sleep consistency: ${(sleepConsistency * 100).toFixed(0)}%.`);

  return {
    estimatedUtcOffset: utcOffset,
    estimatedRegion,
    hasSleepGap,
    longestGapHours: Math.round(longestGapHours * 100) / 100,
    avgSleepGapHours: Math.round(avgSleepGapHours * 100) / 100,
    sleepPeriodsDetected,
    awakePostsPerHour: Math.round(awakePostsPerHour * 100) / 100,
    sleepPostsPerHour: Math.round(sleepPostsPerHour * 100) / 100,
    sleepWakeRatio: Math.round(sleepWakeRatio * 100) / 100,
    sleepConsistency: Math.round(sleepConsistency * 100) / 100,
    confidence,
    summary: parts.join(" "),
  };
}

/**
 * Convenience: format a TimeAnalysis into a text block suitable for
 * inclusion in an LLM prompt used by other detectors.
 */
export function formatTimeAnalysisForPrompt(analysis: TimeAnalysis): string {
  if (analysis.confidence === 0) return "Time analysis: insufficient data.";

  const lines = [
    `=== Time Pattern Analysis (confidence: ${(analysis.confidence * 100).toFixed(0)}%) ===`,
    `Sleep gap detected: ${analysis.hasSleepGap ? "YES" : "NO"}`,
    `Longest gap: ${analysis.longestGapHours}h | Avg daily gap: ${analysis.avgSleepGapHours}h`,
    `Sleep periods found: ${analysis.sleepPeriodsDetected}`,
    `Activity — awake: ${analysis.awakePostsPerHour} posts/h, sleep window: ${analysis.sleepPostsPerHour} posts/h`,
    `Sleep/wake ratio: ${analysis.sleepWakeRatio} (0=human-like, 1=flat/bot-like)`,
    `Sleep consistency: ${(analysis.sleepConsistency * 100).toFixed(0)}%`,
  ];

  if (analysis.estimatedUtcOffset !== null) {
    const sign = analysis.estimatedUtcOffset >= 0 ? "+" : "";
    lines.push(`Estimated timezone: UTC${sign}${analysis.estimatedUtcOffset} → ${analysis.estimatedRegion}`);
  }

  lines.push(`Summary: ${analysis.summary}`);
  return lines.join("\n");
}
