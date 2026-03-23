import type { BottleneckAlert, DataSource, MetricSnapshot, PipelineStageInfo, SimulationWorkflow } from "@gpu-sim/shared";

import { PIPELINE_METRIC_LINKS } from "./pipelineLinks";

type RunStatus = "running" | "paused" | "step_once" | "stopped" | "completed";

export type LocalSimulationParams = {
  durationTicks: number;
  tickMs: number;
  seed: number;
  flashAttention: boolean;
  workflow: SimulationWorkflow;
  modelId: string;
  gpuId: string;
  /** Selected source label that gets written into `MetricSnapshot.source`. */
  source: DataSource;
};

export type LocalSimulationCallbacks = {
  onSnapshot: (snapshot: MetricSnapshot) => void;
  onAlert?: (alert: BottleneckAlert) => void;
  onDone?: () => void;
  onStatus?: (status: RunStatus) => void;
};

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function stackSelectionTune(snapshot: MetricSnapshot): number {
  const mh = [...snapshot.modelId].reduce((a, c) => a + c.charCodeAt(0), 0);
  const gh = [...snapshot.gpuId].reduce((a, c) => a + c.charCodeAt(0), 0);
  const wf = snapshot.workflow ?? "training";
  let w = 0;
  if (wf === "inference") w = -0.035;
  if (wf === "throughput_benchmark") w = 0.045;
  return 1 + (mh % 23) / 220 - (gh % 19) / 240 + w;
}

async function sleep(ms: number) {
  await new Promise<void>((r) => setTimeout(() => r(), ms));
}

type SyntheticOptions = {
  durationTicks: number;
  tickMs: number;
  seed: number;
  flashAttention?: boolean;
  workflow?: SimulationWorkflow;
  modelId: string;
  gpuId: string;
  snapshotSource?: DataSource;
};

function generateSyntheticSnapshots(opts: SyntheticOptions): MetricSnapshot[] {
  const rand = mulberry32(opts.seed);
  const source: DataSource = opts.snapshotSource ?? "synthetic";

  const snapshots: MetricSnapshot[] = [];
  const startTimeMs = Date.now();
  const workflow = opts.workflow ?? "training";

  // Pick stable-ish baselines from modelId/gpuId so the UI doesn't look identical.
  const modelHash = [...opts.modelId].reduce((a, c) => a + c.charCodeAt(0), 0);
  const gpuHash = [...opts.gpuId].reduce((a, c) => a + c.charCodeAt(0), 0);

  const baseLoss = workflow === "inference" ? 3.2 : 10 + (modelHash % 20) * 0.25; // ~10..15 for training
  const minLoss = 1.2 + (modelHash % 7) * 0.08; // ~1.2..1.7
  const decay = workflow === "throughput_benchmark" ? 0.999 : 0.997 - ((modelHash % 11) * 0.00015);

  const workflowMfuBias = workflow === "throughput_benchmark" ? 0.08 : workflow === "inference" ? -0.02 : 0;
  const workflowHbmBias = workflow === "inference" ? 0.06 : workflow === "throughput_benchmark" ? -0.03 : 0;
  const workflowOccBias = workflow === "throughput_benchmark" ? 0.05 : 0;

  const baseMfu = 0.36 + (gpuHash % 17) / 100 + (opts.flashAttention ? 0.06 : 0) + workflowMfuBias;
  const baseHbm = 0.55 + (gpuHash % 23) / 200 - (opts.flashAttention ? 0.08 : 0) + workflowHbmBias;
  const baseOcc = 0.5 + (modelHash % 29) / 200 + workflowOccBias;

  for (let i = 0; i < opts.durationTicks; i++) {
    const noise = (rand() - 0.5) * 0.15;
    const loss = Math.max(minLoss, baseLoss * Math.pow(decay, i) + noise);

    const gradNorm = 0.4 + loss * 0.08 + Math.abs(noise) * 0.2;

    const mfu = clamp01(baseMfu + (rand() - 0.5) * 0.03 + Math.sin(i / 120) * 0.01);
    const hbmBandwidthUtil = clamp01(baseHbm + (rand() - 0.5) * 0.05 + Math.cos(i / 90) * 0.015);
    const smOccupancy = clamp01(baseOcc + (rand() - 0.5) * 0.06 + Math.sin(i / 140) * 0.02);

    const warpStallRate = clamp01(0.12 + (1 - smOccupancy) * 0.55 + (hbmBandwidthUtil - 0.6) * 0.12);
    const tensorCoreUtil = clamp01(mfu * 1.28 + (workflow === "throughput_benchmark" ? 0.08 : 0));
    const l2HitRate = clamp01(0.65 - (hbmBandwidthUtil - 0.55) * 0.35 + (rand() - 0.5) * 0.04);
    const flashAttnSavings = clamp01((opts.flashAttention ? 0.16 : 0.04) + (hbmBandwidthUtil - 0.4) * 0.25);
    const kvCacheUtil = clamp01(
      workflow === "inference"
        ? 0.42 + (rand() - 0.5) * 0.08 + Math.sin(i / 40) * 0.04
        : 0.15 + (rand() - 0.5) * 0.05
    );
    const cpuInputUtil = clamp01(
      workflow === "training"
        ? 0.62 + (rand() - 0.5) * 0.08
        : workflow === "throughput_benchmark"
          ? 0.78 + (rand() - 0.5) * 0.06
          : 0.48 + (rand() - 0.5) * 0.07
    );
    const energyWatts = Math.max(140, 210 + mfu * 260 + hbmBandwidthUtil * 90 + (rand() - 0.5) * 18);

    const timestampMs = startTimeMs + i * opts.tickMs;

    snapshots.push({
      tick: i,
      modelId: opts.modelId,
      gpuId: opts.gpuId,
      mfu,
      hbmBandwidthUtil,
      smOccupancy,
      loss,
      gradNorm,
      warpStallRate,
      tensorCoreUtil,
      l2HitRate,
      flashAttnSavings,
      kvCacheUtil,
      cpuInputUtil,
      energyWatts,
      workflow,
      tflopsAchieved: 100 + mfu * 250 + (opts.flashAttention ? 10 : 0),
      tokensPerSec:
        workflow === "throughput_benchmark"
          ? 1600 + mfu * 1300 + (1 - warpStallRate) * 260
          : 1000 + mfu * 800 + (1 - warpStallRate) * 200,
      source,
      timestampMs
    });
  }

  return snapshots;
}

function snapshotSnippet(s: MetricSnapshot, keys: string[], extra?: Record<string, number | string | undefined>) {
  const out: Record<string, number | string | undefined> = { ...(extra ?? {}) };
  for (const k of keys) {
    const v = (s as unknown as Record<string, unknown>)[k];
    if (v !== undefined) out[k] = v as number | string;
  }
  return out;
}

function finalizePipelineTrace(tick: number, stages: PipelineStageInfo[]) {
  return {
    tick,
    stages,
    links: PIPELINE_METRIC_LINKS
  };
}

function detectBottleneck(alertState: { recentLosses: number[] }, snapshot: MetricSnapshot): BottleneckAlert | null {
  if (snapshot.loss != null) {
    alertState.recentLosses.push(snapshot.loss);
    if (alertState.recentLosses.length > 120) alertState.recentLosses.shift();
  }

  const mfu = snapshot.mfu;
  const hbmUtil = snapshot.hbmBandwidthUtil;
  const smOcc = snapshot.smOccupancy;
  const gradNorm = snapshot.gradNorm ?? 0;

  if (mfu < 0.4 && hbmUtil > 0.7) {
    return { type: "memory_bound", mfu, hbmUtil, tick: snapshot.tick };
  }
  if (smOcc < 0.35) {
    return { type: "low_occupancy", smOccupancy: smOcc, tick: snapshot.tick };
  }
  if (gradNorm > 10.0) {
    return { type: "gradient_explosion", gradNorm, tick: snapshot.tick };
  }

  if (snapshot.loss != null && alertState.recentLosses.length >= 60) {
    const lastN = 20;
    const prevN = 20;
    const recent = avg(alertState.recentLosses.slice(-lastN));
    const prev = avg(alertState.recentLosses.slice(-(lastN + prevN), -lastN));
    const denom = Math.max(1e-6, prev);
    const recentLossDelta = (recent - prev) / denom;

    if (Math.abs(recentLossDelta) < 0.01) {
      return { type: "loss_plateau", recentLossDelta, tick: snapshot.tick };
    }
  }

  return null;
}

function avg(xs: number[]) {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function processPipeline(baseSnapshot: MetricSnapshot, tick: number, state: { alertState: { recentLosses: number[] } }) {
  const working: MetricSnapshot = { ...baseSnapshot, tick };

  // Stage 1 — model.
  const prevLoss = (state.alertState as any).prevLoss as number | undefined;
  const modelRes = {
    loss: working.loss ?? working.loss,
    gradNorm: working.gradNorm ?? working.gradNorm,
    convergenceRate: prevLoss == null ? undefined : Math.max(0, ((prevLoss - (working.loss ?? prevLoss)) / Math.max(1e-6, prevLoss)))
  };
  // Store for next tick for convergenceRate (kept for completeness; UI may not display it).
  (state.alertState as any).prevLoss = modelRes.loss;

  let staged = {
    ...working,
    loss: modelRes.loss,
    gradNorm: modelRes.gradNorm
  };

  const stages: PipelineStageInfo[] = [];
  stages.push({
    order: 1,
    moduleId: "model",
    title: "Model & gradients",
    summary: "Loss and gradient norms (replay or synthetic) set the optimization signal for this tick.",
    readsMetrics: [],
    producesMetrics: ["loss", "gradNorm"],
    valuesAtStage: snapshotSnippet(staged, ["loss", "gradNorm"])
  });

  // Stage 2 — compute.
  const baseMfu = staged.mfu ?? 0.45;
  const grad = staged.gradNorm ?? 1;
  const tune = stackSelectionTune(staged);
  const mfu = clamp01(baseMfu * (0.82 + 0.05 * Math.min(grad, 5)) * tune);
  const tensorCoreUtil = clamp01(mfu * 1.25);
  const tflopsAchieved = staged.tflopsAchieved ?? 100 + mfu * 250;
  staged = { ...staged, mfu, tflopsAchieved, tensorCoreUtil };
  stages.push({
    order: 2,
    moduleId: "compute",
    title: "Compute / GEMM",
    summary: "MFU and tensor-core utilization absorb gradient-heavy matmul work from the model stage.",
    readsMetrics: ["loss", "gradNorm"],
    producesMetrics: ["mfu", "tflopsAchieved", "tensorCoreUtil"],
    valuesAtStage: snapshotSnippet(staged, ["mfu", "tflopsAchieved", "tensorCoreUtil"])
  });

  // Stage 3 — memory.
  const baseHbm = staged.hbmBandwidthUtil ?? 0.65;
  const mfuForMem = staged.mfu ?? 0.5;
  const tc = staged.tensorCoreUtil ?? clamp01(mfuForMem * 1.25);
  const tuneMem = stackSelectionTune(staged);

  const hbmBandwidthUtil = clamp01(
    (0.28 + mfuForMem * 0.38 + tc * 0.2 + (baseHbm - 0.5) * 0.22) * (0.94 + (tuneMem - 1) * 0.35)
  );

  const l2HitRate =
    staged.source === "synthetic"
      ? clamp01(0.32 + (0.64 - hbmBandwidthUtil) * 0.42)
      : clamp01(0.28 + (0.72 - hbmBandwidthUtil) * 0.32);

  const flashAttnSavings = clamp01((0.62 - l2HitRate) * 0.38 + (mfuForMem - 0.35) * 0.2 + (hbmBandwidthUtil - 0.42) * 0.14);

  staged = { ...staged, hbmBandwidthUtil, l2HitRate, flashAttnSavings };
  stages.push({
    order: 3,
    moduleId: "memory",
    title: "Memory & caches",
    summary: "HBM traffic and L2 behavior follow compute intensity; flash-attn savings reflect IO-aware patterns.",
    readsMetrics: ["mfu", "tensorCoreUtil"],
    producesMetrics: ["hbmBandwidthUtil", "l2HitRate", "flashAttnSavings"],
    valuesAtStage: snapshotSnippet(staged, ["hbmBandwidthUtil", "l2HitRate", "flashAttnSavings"])
  });

  // Stage 4 — warp scheduler.
  const hbm = staged.hbmBandwidthUtil ?? 0.65;
  const mfuForWarp = staged.mfu ?? 0.5;
  const l2 = staged.l2HitRate ?? 0.5;
  const tuneWarp = stackSelectionTune(staged);

  const smOccupancy = clamp01(
    (0.2 + mfuForWarp * 0.44 - Math.max(0, hbm - 0.52) * 0.36 + (l2 - 0.38) * 0.14) *
      (0.96 + (tuneWarp - 1) * 0.25)
  );

  const warpStallRate = clamp01(0.1 + (hbm - 0.38) * 0.52 + (1 - l2) * 0.24 + (1 - tuneWarp) * 0.04);
  const activeWarps = Math.round(32 * smOccupancy * 2.0);

  staged = { ...staged, smOccupancy, warpStallRate, activeWarps };
  stages.push({
    order: 4,
    moduleId: "warp_scheduler",
    title: "Warp scheduler",
    summary: "SM occupancy and warp stalls close the loop between memory latency and runnable warps.",
    readsMetrics: ["hbmBandwidthUtil", "l2HitRate", "mfu"],
    producesMetrics: ["smOccupancy", "warpStallRate"],
    valuesAtStage: snapshotSnippet(staged, ["smOccupancy", "warpStallRate"], { activeWarps })
  });

  const pipelineTrace = finalizePipelineTrace(tick, stages);
  const out: MetricSnapshot = { ...staged, pipelineTrace };
  const alert = detectBottleneck(state.alertState, out);
  return { snapshot: out, alert };
}

export class LocalSimulationRunner {
  private status: RunStatus = "running";
  private cancelled = false;
  private currentTick = 0;

  private snapshots: MetricSnapshot[] = [];
  private alertState: { recentLosses: number[]; prevLoss?: number } = { recentLosses: [] };

  private onSnapshot: (s: MetricSnapshot) => void = () => undefined;
  private onAlert: ((a: BottleneckAlert) => void) | undefined;
  private onDone: (() => void) | undefined;
  private onStatus: ((st: RunStatus) => void) | undefined;

  constructor(private params: LocalSimulationParams) {}

  start(callbacks: LocalSimulationCallbacks) {
    this.onSnapshot = callbacks.onSnapshot;
    this.onAlert = callbacks.onAlert;
    this.onDone = callbacks.onDone;
    this.onStatus = callbacks.onStatus;

    this.cancelled = false;
    this.currentTick = 0;
    this.status = "running";
    this.alertState = { recentLosses: [] };

    this.snapshots = generateSyntheticSnapshots({
      durationTicks: this.params.durationTicks,
      tickMs: this.params.tickMs,
      seed: this.params.seed,
      flashAttention: this.params.flashAttention,
      workflow: this.params.workflow,
      modelId: this.params.modelId,
      gpuId: this.params.gpuId,
      snapshotSource: this.params.source
    });

    this.onStatus?.(this.status);
    this.loop();
  }

  pause() {
    if (this.status === "running") {
      this.status = "paused";
      this.onStatus?.(this.status);
    }
  }

  resume() {
    if (this.status === "paused") {
      this.status = "running";
      this.onStatus?.(this.status);
    }
  }

  stepOnce() {
    if (this.status === "paused") {
      this.status = "step_once";
      this.onStatus?.(this.status);
    }
  }

  stop() {
    this.cancelled = true;
    this.status = "stopped";
    this.onStatus?.(this.status);
  }

  private async loop() {
    const { tickMs, durationTicks } = this.params;

    while (!this.cancelled && this.currentTick < durationTicks) {
      if (this.status === "paused") {
        await sleep(20);
        continue;
      }

      const tick = this.currentTick;
      const base = this.snapshots[tick];
      const { snapshot, alert } = processPipeline(base, tick, { alertState: this.alertState });

      this.onSnapshot(snapshot);
      if (alert && this.onAlert) this.onAlert(alert);

      this.currentTick++;

      if (this.status === "step_once") {
        this.status = "paused";
        this.onStatus?.(this.status);
      }

      if (this.status === "running") {
        // Keep the local runner responsive by respecting the requested tick speed.
        await sleep(Math.max(1, tickMs));
      }
    }

    if (!this.cancelled) {
      this.status = "completed";
      this.onStatus?.(this.status);
      this.onDone?.();
    }
  }
}

