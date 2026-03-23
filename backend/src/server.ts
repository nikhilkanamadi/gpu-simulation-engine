import http from "node:http";
import path from "node:path";
import { WebSocketServer } from "ws";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { z } from "zod";

import { DB_FILE, DEFAULT_DURATION_TICKS, DEFAULT_TICK_MS, PORT, WS_PATH } from "./config";
import { TimeSeriesStore } from "./storage/TimeSeriesStore";
import { LiveWebSocketHub } from "./ws/LiveWebSocketHub";
import { RunManager } from "./runs/RunManager";
import type { StartRunRequest } from "@gpu-sim/shared";

dotenv.config();

const StartRunRequestSchema = z.object({
  modelId: z.string().min(1),
  gpuId: z.string().min(1),
  source: z.enum(["mlperf", "kineto", "hta", "wandb", "lambda", "synthetic"]),
  options: z
    .object({
      durationTicks: z.number().int().positive().optional().default(DEFAULT_DURATION_TICKS),
      tickMs: z.number().int().positive().optional().default(DEFAULT_TICK_MS),
      seed: z.number().int().optional(),
      flashAttention: z.boolean().optional(),
      workflow: z.enum(["training", "inference", "throughput_benchmark"]).optional()
    })
    .optional()
});

async function main() {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true });
  });

  const store = await TimeSeriesStore.create(DB_FILE);
  const wsHub = new LiveWebSocketHub();
  const runManager = new RunManager(store, wsHub);

  app.get("/api/runs", (_req, res) => {
    res.json(runManager.listRuns());
  });

  app.post("/api/runs/start", async (req, res) => {
    try {
      const body = StartRunRequestSchema.parse(req.body) as StartRunRequest;
      const runState = await runManager.startRun(body);
      res.json({ runId: runState.runId });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      res.status(400).json({ error: message });
    }
  });

  app.post("/api/runs/:runId/pause", (req, res) => {
    runManager.pause(req.params.runId);
    res.json({ ok: true });
  });

  app.post("/api/runs/:runId/resume", (req, res) => {
    runManager.resume(req.params.runId);
    res.json({ ok: true });
  });

  app.post("/api/runs/:runId/step", (req, res) => {
    runManager.stepOnce(req.params.runId);
    res.json({ ok: true });
  });

  app.post("/api/runs/:runId/speed", (req, res) => {
    const speed = Number(req.body?.speedMultiplier ?? 1.0);
    runManager.setSpeed(req.params.runId, speed);
    res.json({ ok: true });
  });

  app.get("/api/runs/:runId/snapshots", (req, res) => {
    const runId = req.params.runId;
    const offset = Number(req.query.offset ?? 0);
    const limit = Math.min(500, Number(req.query.limit ?? 200));
    const rows = store.getSnapshots(runId, offset, limit);
    res.json({ rows });
  });

  app.get("/api/runs/:runId/summary", (req, res) => {
    const runId = req.params.runId;
    res.json(store.getSummary(runId));
  });

  // Create HTTP + WS.
  const httpServer = http.createServer(app);
  const wss = new WebSocketServer({ server: httpServer, path: WS_PATH });

  wss.on("connection", (ws, request) => {
    try {
      const url = new URL(request.url ?? "", `http://localhost:${PORT}`);
      const runId = url.searchParams.get("runId") ?? "";
      if (!runId) {
        ws.close(1008, "Missing runId query param");
        return;
      }
      wsHub.addClient(runId, ws);

      ws.on("close", () => wsHub.removeClient(runId, ws));
      ws.on("error", () => wsHub.removeClient(runId, ws));
    } catch {
      ws.close(1008, "Bad connection URL");
    }
  });

  httpServer.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`Backend listening on http://localhost:${PORT}`);
    // eslint-disable-next-line no-console
    console.log(`WebSocket: ws://localhost:${PORT}${WS_PATH}?runId=...`);
  });
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});

