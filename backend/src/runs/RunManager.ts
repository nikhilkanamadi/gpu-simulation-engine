import crypto from "node:crypto";
import type { MetricSnapshot, StartRunRequest } from "@gpu-sim/shared";
import type { TimeSeriesStore } from "../storage/TimeSeriesStore";
import { Normalizer } from "../ingestion/Normalizer";
import { LiveWebSocketHub } from "../ws/LiveWebSocketHub";
import { Orchestrator } from "../simulation/Orchestrator";
import { sleep } from "../utils/sleep";

type RunStatus = "running" | "paused" | "step_once" | "stopped" | "completed";

export interface RunState {
  runId: string;
  modelId: string;
  gpuId: string;
  source: StartRunRequest["source"];
  startTimeMs: number;
  totalTicks: number;
  currentTick: number;
  status: RunStatus;
  speedMultiplier: number;
}

export class RunManager {
  private runs = new Map<string, RunState>();
  private orchestrators = new Map<string, Orchestrator>();
  private snapshotsByRunId = new Map<string, MetricSnapshot[]>();
  private runLoops = new Map<string, Promise<void>>();

  constructor(
    private readonly store: TimeSeriesStore,
    private readonly wsHub: LiveWebSocketHub
  ) {}

  listRuns() {
    return Array.from(this.runs.values()).map((r) => ({
      runId: r.runId,
      modelId: r.modelId,
      gpuId: r.gpuId,
      source: r.source,
      startTimeMs: r.startTimeMs,
      endTimeMs: r.status === "completed" ? Date.now() : undefined,
      totalTicks: r.totalTicks
    }));
  }

  async startRun(req: StartRunRequest): Promise<RunState> {
    const runId = crypto.randomUUID();

    const durationTicks = req.options?.durationTicks ?? 1000;
    const tickMs = req.options?.tickMs ?? 50;
    const speedMultiplier = 1.0;

    const snapshots = Normalizer.normalize(req);
    if (snapshots.length !== durationTicks) {
      // Keep the run deterministic even if the generator chooses a different default.
      snapshots.length = durationTicks;
    }

    const orchestrator = new Orchestrator({ tickMs, durationTicks });
    orchestrator.reset();

    const runState: RunState = {
      runId,
      modelId: req.modelId,
      gpuId: req.gpuId,
      source: req.source,
      startTimeMs: Date.now(),
      totalTicks: durationTicks,
      currentTick: 0,
      status: "running",
      speedMultiplier
    };

    this.runs.set(runId, runState);
    this.orchestrators.set(runId, orchestrator);
    this.snapshotsByRunId.set(runId, snapshots);

    const loop = this.runLoop(runId, tickMs).catch((err) => {
      // eslint-disable-next-line no-console
      console.error(`Run loop failed for ${runId}`, err);
      const state = this.runs.get(runId);
      if (state) state.status = "stopped";
    });
    this.runLoops.set(runId, loop);

    return runState;
  }

  pause(runId: string) {
    const state = this.requireRun(runId);
    if (state.status === "running") state.status = "paused";
  }

  resume(runId: string) {
    const state = this.requireRun(runId);
    if (state.status === "paused") state.status = "running";
  }

  stepOnce(runId: string) {
    const state = this.requireRun(runId);
    if (state.status === "paused") state.status = "step_once";
  }

  setSpeed(runId: string, speedMultiplier: number) {
    const state = this.requireRun(runId);
    state.speedMultiplier = Math.max(0.1, speedMultiplier);
  }

  async stop(runId: string) {
    const state = this.requireRun(runId);
    state.status = "stopped";
    // Attempt a final persist for convenience.
    this.store.persist();
  }

  getRun(runId: string): RunState | undefined {
    return this.runs.get(runId);
  }

  private requireRun(runId: string): RunState {
    const state = this.runs.get(runId);
    if (!state) throw new Error(`Unknown runId: ${runId}`);
    return state;
  }

  private async runLoop(runId: string, tickMs: number) {
    const orchestrator = this.orchestrators.get(runId);
    const snapshots = this.snapshotsByRunId.get(runId);
    const state = this.runs.get(runId);
    if (!orchestrator || !snapshots || !state) return;

    let inserted = 0;
    while (state.currentTick < state.totalTicks) {
      if (state.status === "stopped") return;
      if (state.status === "paused") {
        await sleep(20);
        continue;
      }

      const tickIndex = state.currentTick;
      const base = snapshots[tickIndex];
      const { snapshot, alert } = orchestrator.processTick(base, tickIndex);

      // Persist + broadcast.
      this.store.insertSnapshot(runId, snapshot);
      inserted++;
      if (inserted % 50 === 0) this.store.persist();

      this.wsHub.broadcastSnapshot(runId, snapshot);
      if (alert) this.wsHub.broadcastAlert(runId, alert);

      state.currentTick++;

      if (state.status === "step_once") state.status = "paused";
      if (state.currentTick >= state.totalTicks) break;

      if (state.status === "running") {
        const wait = tickMs / state.speedMultiplier;
        await sleep(wait);
      }
    }

    state.status = "completed";
    this.store.persist();
  }
}

