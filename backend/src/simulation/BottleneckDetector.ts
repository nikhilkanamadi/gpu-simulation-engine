import type { BottleneckAlert, MetricSnapshot } from "@gpu-sim/shared";

export class BottleneckDetector {
  private recentLosses: number[] = [];
  private readonly maxWindow = 120;

  reset() {
    this.recentLosses = [];
  }

  detect(snapshot: MetricSnapshot): BottleneckAlert | null {
    if (snapshot.loss != null) {
      this.recentLosses.push(snapshot.loss);
      if (this.recentLosses.length > this.maxWindow) this.recentLosses.shift();
    }

    const mfu = snapshot.mfu;
    const hbmUtil = snapshot.hbmBandwidthUtil;
    const smOcc = snapshot.smOccupancy;
    const gradNorm = snapshot.gradNorm ?? 0;

    // Prioritize memory/occupancy/instability signals over loss plateau for MVP clarity.
    if (mfu < 0.4 && hbmUtil > 0.7) {
      return { type: "memory_bound", mfu, hbmUtil, tick: snapshot.tick };
    }
    if (smOcc < 0.35) {
      return { type: "low_occupancy", smOccupancy: smOcc, tick: snapshot.tick };
    }
    if (gradNorm > 10.0) {
      return { type: "gradient_explosion", gradNorm, tick: snapshot.tick };
    }

    // Loss plateau: compare trailing average vs earlier average.
    if (snapshot.loss != null && this.recentLosses.length >= 60) {
      const lastN = 20;
      const prevN = 20;
      const recent = avg(this.recentLosses.slice(-lastN));
      const prev = avg(this.recentLosses.slice(-(lastN + prevN), -lastN));
      const denom = Math.max(1e-6, prev);
      const recentLossDelta = (recent - prev) / denom;

      if (Math.abs(recentLossDelta) < 0.01) {
        return { type: "loss_plateau", recentLossDelta, tick: snapshot.tick };
      }
    }

    return null;
  }
}

function avg(xs: number[]) {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

