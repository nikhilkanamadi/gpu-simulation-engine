import fs from "node:fs";
import path from "node:path";
import type { DataSource, MetricSnapshot, StartRunRequest } from "@gpu-sim/shared";

import { generateSyntheticSnapshots } from "./synthetic/SyntheticGenerator";

export type IngestionRequest = StartRunRequest;

export class Normalizer {
  static normalize(request: IngestionRequest): MetricSnapshot[] {
    const source: DataSource = request.source;
    const durationTicks = request.options?.durationTicks ?? Number(process.env.DURATION_TICKS ?? 1000);
    const tickMs = request.options?.tickMs ?? Number(process.env.TICK_MS ?? 50);
    const seed = request.options?.seed ?? 42;
    const flashAttention = request.options?.flashAttention ?? true;
    const workflow = request.options?.workflow ?? "training";

    const syntheticStandIn = (snapshotSource: DataSource) =>
      generateSyntheticSnapshots({
        durationTicks,
        tickMs,
        seed,
        flashAttention,
        workflow,
        modelId: request.modelId,
        gpuId: request.gpuId,
        snapshotSource
      });

    if (source === "synthetic") {
      return syntheticStandIn("synthetic");
    }

    if (source === "mlperf") {
      const mlperfPath = path.join(process.cwd(), "data", "mlperf", "results-v4.1.json");
      if (!fs.existsSync(mlperfPath)) {
        // eslint-disable-next-line no-console
        console.warn(`[Normalizer] MLPerf file missing at ${mlperfPath}; using synthetic stand-in tagged as mlperf.`);
        return syntheticStandIn("mlperf");
      }

      try {
        const raw = fs.readFileSync(mlperfPath, "utf-8");
        const parsed = JSON.parse(raw) as unknown;

        const entry =
          (parsed as any)?.model ?? (Array.isArray(parsed) ? (parsed as any[])[0] : undefined) ?? parsed;

        const samplesPerSec = Number((entry as any)?.samples_per_second ?? 0);
        const lossProxy = Number((entry as any)?.time_to_train_minutes ?? 30);

        const mfuBase = clamp01(samplesPerSec / 25000);
        const baseLoss = 10 + clamp01(lossProxy / 120) * 5;

        const snapshots: MetricSnapshot[] = [];
        const startTimeMs = Date.now();
        const noiseRand = mulberry32(seed + 1337);
        for (let tick = 0; tick < durationTicks; tick++) {
          const decay = Math.pow(0.997, tick);
          const noise = (noiseRand() - 0.5) * 0.18;
          const loss = Math.max(1.0, baseLoss * decay + noise);
          const gradNorm = 0.35 + loss * 0.07;

          const mfu = clamp01(mfuBase + (noiseRand() - 0.5) * 0.05);
          const hbmBandwidthUtil = clamp01(0.68 - (mfu - 0.4) * 0.2 + (noiseRand() - 0.5) * 0.04);
          const smOccupancy = clamp01(0.58 + (mfu - 0.4) * 0.15 + (noiseRand() - 0.5) * 0.05);
          const warpStallRate = clamp01(0.14 + (1 - smOccupancy) * 0.5 + (hbmBandwidthUtil - 0.6) * 0.12);

          snapshots.push({
            tick,
            modelId: request.modelId,
            gpuId: request.gpuId,
            mfu,
            hbmBandwidthUtil,
            smOccupancy,
            tflopsAchieved: 100 + mfu * 250,
            loss,
            gradNorm,
            tokensPerSec: 1000 + mfu * 800,
            warpStallRate,
            tensorCoreUtil: clamp01(mfu * 1.25),
            l2HitRate: clamp01(0.62 - (hbmBandwidthUtil - 0.55) * 0.32),
            flashAttnSavings: clamp01((flashAttention ? 0.15 : 0.05) + (hbmBandwidthUtil - 0.45) * 0.2),
            kvCacheUtil: clamp01(0.18 + (noiseRand() - 0.5) * 0.04),
            cpuInputUtil: clamp01(0.65 + (noiseRand() - 0.5) * 0.08),
            energyWatts: Math.max(140, 220 + mfu * 230 + hbmBandwidthUtil * 85),
            workflow,
            source: "mlperf",
            timestampMs: startTimeMs + tick * tickMs
          });
        }

        return snapshots;
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[Normalizer] MLPerf parse failed; using synthetic stand-in.", err);
        return syntheticStandIn("mlperf");
      }
    }

    // kineto / hta / wandb / lambda: no file ingestion in MVP — deterministic synthetic tagged by source.
    // eslint-disable-next-line no-console
    console.warn(`[Normalizer] Source "${source}" has no trace ingest yet; using synthetic stand-in.`);
    return syntheticStandIn(source);
  }
}

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

