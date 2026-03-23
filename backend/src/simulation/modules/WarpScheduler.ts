import type { AlgorithmModule, SchedulerResult, SimTick } from "@gpu-sim/shared";

import { stackSelectionTune } from "../selectionTune";

export class WarpScheduler implements AlgorithmModule<SchedulerResult> {
  name = "warp_scheduler";

  reset(): void {
    // stateless in MVP
  }

  step(tick: SimTick): SchedulerResult {
    const snapshot = tick.snapshot;
    const hbm = snapshot.hbmBandwidthUtil ?? 0.65;
    const mfu = snapshot.mfu ?? 0.5;
    const l2 = snapshot.l2HitRate ?? 0.5;
    const tune = stackSelectionTune(snapshot);
    // Chained: occupancy balances compute vs memory latency hiding; stalls track DRAM pressure.
    const smOccupancy = clamp01(
      (0.2 + mfu * 0.44 - Math.max(0, hbm - 0.52) * 0.36 + (l2 - 0.38) * 0.14) * (0.96 + (tune - 1) * 0.25)
    );

    const warpStallRate = clamp01(0.1 + (hbm - 0.38) * 0.52 + (1 - l2) * 0.24 + (1 - tune) * 0.04);

    const activeWarps = Math.round(32 * smOccupancy * 2.0);

    return {
      smOccupancy,
      warpStallRate,
      activeWarps
    };
  }
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

