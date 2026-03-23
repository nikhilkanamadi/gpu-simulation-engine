import type { MetricLink } from "@gpu-sim/shared";

/** Kept in sync with `packages/shared/src/types.ts` (`PIPELINE_METRIC_LINKS`). */
export const PIPELINE_METRIC_LINKS: MetricLink[] = [
  {
    fromKey: "loss",
    toKey: "gradNorm",
    description: "Backprop couples the loss surface to gradient magnitude."
  },
  {
    fromKey: "gradNorm",
    toKey: "mfu",
    description: "Gradient-heavy steps drive GEMM/tensor work, shaping model FLOP utilization."
  },
  {
    fromKey: "mfu",
    toKey: "tensorCoreUtil",
    description: "When kernels map to MMA, tensor cores track the compute-heavy portion of MFU."
  },
  {
    fromKey: "mfu",
    toKey: "hbmBandwidthUtil",
    description: "Higher arithmetic intensity increases traffic for weights, activations, and gradients."
  },
  {
    fromKey: "hbmBandwidthUtil",
    toKey: "l2HitRate",
    description: "Memory pressure competes with on-chip cache residency."
  },
  {
    fromKey: "l2HitRate",
    toKey: "flashAttnSavings",
    description: "IO-aware attention benefits when locality and block structure reduce HBM round-trips."
  },
  {
    fromKey: "hbmBandwidthUtil",
    toKey: "warpStallRate",
    description: "DRAM latency and queueing stall warps waiting on global loads."
  },
  {
    fromKey: "hbmBandwidthUtil",
    toKey: "smOccupancy",
    description: "Memory-bound waves often launch with enough warps to hide latency—or thin out if stalled."
  },
  {
    fromKey: "mfu",
    toKey: "smOccupancy",
    description: "Compute-bound tiles can keep SMs busy when not latency-limited."
  }
];
