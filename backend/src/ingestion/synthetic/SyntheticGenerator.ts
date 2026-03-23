import type { DataSource, MetricSnapshot, SimulationWorkflow } from "@gpu-sim/shared";

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SyntheticOptions {
  durationTicks: number;
  tickMs: number;
  seed: number;
  flashAttention?: boolean;
  workflow?: SimulationWorkflow;
  modelId: string;
  gpuId: string;
  /** When set, snapshots use this `source` (e.g. stand-in for mlperf/kineto when no trace file). */
  snapshotSource?: DataSource;
}

export function generateSyntheticSnapshots(opts: SyntheticOptions): MetricSnapshot[] {
  const rand = mulberry32(opts.seed);
  const source: DataSource = opts.snapshotSource ?? "synthetic";

  const snapshots: MetricSnapshot[] = [];
  const startTimeMs = Date.now();
  const workflow = opts.workflow ?? "training";

  // Pick stable-ish baselines from modelId/gpuId so the UI doesn't look identical for all choices.
  const modelHash = [...opts.modelId].reduce((a, c) => a + c.charCodeAt(0), 0);
  const gpuHash = [...opts.gpuId].reduce((a, c) => a + c.charCodeAt(0), 0);

  const baseLoss = workflow === "inference" ? 3.2 : 10 + (modelHash % 20) * 0.25; // ~10..15 for training
  const minLoss = 1.2 + (modelHash % 7) * 0.08; // ~1.2..1.7
  const decay =
    workflow === "throughput_benchmark"
      ? 0.999
      : 0.997 - ((modelHash % 11) * 0.00015); // ~0.997 down to ~0.995

  // Compute-like values
  const workflowMfuBias = workflow === "throughput_benchmark" ? 0.08 : workflow === "inference" ? -0.02 : 0;
  const workflowHbmBias = workflow === "inference" ? 0.06 : workflow === "throughput_benchmark" ? -0.03 : 0;
  const workflowOccBias = workflow === "throughput_benchmark" ? 0.05 : 0;
  const baseMfu = 0.36 + (gpuHash % 17) / 100 + (opts.flashAttention ? 0.06 : 0) + workflowMfuBias;
  const baseHbm = 0.55 + (gpuHash % 23) / 200 - (opts.flashAttention ? 0.08 : 0) + workflowHbmBias;
  const baseOcc = 0.5 + (modelHash % 29) / 200 + workflowOccBias;

  for (let i = 0; i < opts.durationTicks; i++) {
    // Exponential loss decay + a little noise.
    const noise = (rand() - 0.5) * 0.15;
    const loss = Math.max(minLoss, baseLoss * Math.pow(decay, i) + noise);

    // Grad norm correlates with loss magnitude for an intuitive dashboard.
    const gradNorm = 0.4 + loss * 0.08 + Math.abs(noise) * 0.2;

    // Efficiency signals have mild drift over time.
    const mfu = clamp01(baseMfu + (rand() - 0.5) * 0.03 + Math.sin(i / 120) * 0.01);
    const hbmBandwidthUtil = clamp01(
      baseHbm + (rand() - 0.5) * 0.05 + Math.cos(i / 90) * 0.015
    );
    const smOccupancy = clamp01(baseOcc + (rand() - 0.5) * 0.06 + Math.sin(i / 140) * 0.02);

    // Higher util and lower occupancy tends to increase stalls.
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
      // Optional fields: keep them populated for richer UI.
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

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

