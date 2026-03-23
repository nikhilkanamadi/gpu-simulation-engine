import type { AlgorithmModule, ModelResult, SimTick } from "@gpu-sim/shared";

export class ModelModule implements AlgorithmModule<ModelResult> {
  name = "model";
  private lastLoss: number | null = null;

  reset(): void {
    this.lastLoss = null;
  }

  step(tick: SimTick): ModelResult {
    const snapshot = tick.snapshot;

    const prevLoss = this.lastLoss;
    let loss = snapshot.loss;
    let gradNorm = snapshot.gradNorm;

    // MVP: if missing from ingestion, synthesize from tick index.
    if (loss == null) {
      const base = 10;
      const decay = 0.997;
      const noise = Math.sin(tick.tick / 23) * 0.1;
      loss = Math.max(0.9, base * Math.pow(decay, tick.tick) + noise);
    }
    if (gradNorm == null) {
      gradNorm = 0.35 + loss * 0.07;
    }

    const convergenceRate =
      prevLoss == null ? undefined : Math.max(0, (prevLoss - loss) / Math.max(1e-6, prevLoss));

    this.lastLoss = loss;

    return {
      loss,
      gradNorm,
      convergenceRate
    };
  }
}

