import type { AlgorithmModule, MemoryResult, SimTick } from "@gpu-sim/shared";

import { stackSelectionTune } from "../selectionTune";

export class MemoryModule implements AlgorithmModule<MemoryResult> {
  name = "memory";

  reset(): void {
    // stateless in MVP
  }

  step(tick: SimTick): MemoryResult {
    const snapshot = tick.snapshot;
    const baseHbm = snapshot.hbmBandwidthUtil ?? 0.65;
    const mfu = snapshot.mfu ?? 0.5;
    const tc = snapshot.tensorCoreUtil ?? clamp01(mfu * 1.25);
    const tune = stackSelectionTune(snapshot);
    // Chained: GEMM-heavy phases raise HBM traffic; selection tune shifts the memory curve.
    const hbmBandwidthUtil = clamp01(
      (0.28 + mfu * 0.38 + tc * 0.2 + (baseHbm - 0.5) * 0.22) * (0.94 + (tune - 1) * 0.35)
    );

    const l2HitRate =
      snapshot.source === "synthetic"
        ? clamp01(0.32 + (0.64 - hbmBandwidthUtil) * 0.42)
        : clamp01(0.28 + (0.72 - hbmBandwidthUtil) * 0.32);

    // IO-aware attention savings track locality and memory pressure (warp stage not run yet).
    const flashAttnSavings = clamp01(
      (0.62 - l2HitRate) * 0.38 + (mfu - 0.35) * 0.2 + (hbmBandwidthUtil - 0.42) * 0.14
    );

    return {
      hbmBandwidthUtil,
      l2HitRate,
      flashAttnSavings
    };
  }
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

