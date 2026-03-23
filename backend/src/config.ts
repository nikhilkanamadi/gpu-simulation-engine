import path from "node:path";

export const PORT = Number(process.env.PORT ?? 3001);

// Persisted SQLite DB for the time-series store.
// Stored inside backend/ for developer convenience.
export const DB_FILE = path.join(process.cwd(), "data", "metrics.sqlite");

export const WS_PATH = "/ws/live";

export const DEFAULT_TICK_MS = Number(process.env.TICK_MS ?? 50);

// Keep synthetic defaults stable and deterministic-ish.
export const DEFAULT_DURATION_TICKS = Number(process.env.DURATION_TICKS ?? 1000);

