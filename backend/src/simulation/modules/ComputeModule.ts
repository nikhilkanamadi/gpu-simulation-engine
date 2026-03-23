import type { AlgorithmModule, ComputeResult, SimTick } from "@gpu-sim/shared";

import { stackSelectionTune } from "../selectionTune";

export class ComputeModule implements AlgorithmModule<ComputeResult> {
  name = "compute";

  reset(): void {
    // stateless in MVP
  }

  step(tick: SimTick): ComputeResult {
    const snapshot = tick.snapshot;
    const baseMfu = snapshot.mfu ?? 0.45;
    const grad = snapshot.gradNorm ?? 1;
    const tune = stackSelectionTune(snapshot);
    // Chained: backward/forward work scales with gradient magnitude; model/gpu/workflow tune the stack.
    const mfu = clamp01(baseMfu * (0.82 + 0.05 * Math.min(grad, 5)) * tune);

    const precisionBoost = tick.snapshot.source === "synthetic" ? 1.0 : 1.0;
    const tensorCoreUtil = clamp01(mfu * 1.25 * precisionBoost);

    const tflopsAchieved = snapshot.tflopsAchieved ?? 100 + mfu * 250;

    return {
      mfu,
      tflopsAchieved,
      tensorCoreUtil
    };
  }
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

