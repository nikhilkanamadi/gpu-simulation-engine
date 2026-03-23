export type DataSource = "mlperf" | "kineto" | "hta" | "wandb" | "lambda" | "synthetic";
export type SimulationWorkflow = "training" | "inference" | "throughput_benchmark";

export type PipelineModuleId = "model" | "compute" | "memory" | "warp_scheduler";

/** Directed explanation between two metric keys on a snapshot. */
export interface MetricLink {
  fromKey: string;
  toKey: string;
  description: string;
}

/** One row in the per-tick pipeline trace (algorithms run in `order`). */
export interface PipelineStageInfo {
  order: number;
  moduleId: PipelineModuleId;
  title: string;
  summary: string;
  /** Metric keys this stage primarily reads from upstream state. */
  readsMetrics: string[];
  /** Metric keys this stage writes or refines. */
  producesMetrics: string[];
  /** Snapshot fields after this stage completes (subset). */
  valuesAtStage: Record<string, number | string | undefined>;
}

/** Emitted each tick so the UI can show ordered execution and metric coupling. */
export interface PipelineTrace {
  tick: number;
  stages: PipelineStageInfo[];
  links: MetricLink[];
}

export interface MetricSnapshot {
  tick: number;
  modelId: string;
  gpuId: string;
  mfu: number; // 0..1
  hbmBandwidthUtil: number; // 0..1
  smOccupancy: number; // 0..1
  tflopsAchieved?: number;
  loss?: number;
  gradNorm?: number;
  tokensPerSec?: number;
  warpStallRate?: number;
  tensorCoreUtil?: number;
  l2HitRate?: number;
  flashAttnSavings?: number;
  kvCacheUtil?: number;
  energyWatts?: number;
  cpuInputUtil?: number;
  workflow?: SimulationWorkflow;
  source: DataSource;
  timestampMs: number;
  /** Present when produced by the chained Orchestrator (live runs). */
  pipelineTrace?: PipelineTrace;
}

export interface SimTick {
  tick: number;
  snapshot: MetricSnapshot;
  clockMs: number;
}

export interface AlgorithmModule<TResult> {
  name: string;
  step(tick: SimTick): TResult;
  reset(): void;
}

export interface ModelResult {
  loss?: number;
  gradNorm?: number;
  convergenceRate?: number;
}

export interface ComputeResult {
  mfu: number;
  tflopsAchieved?: number;
  tensorCoreUtil?: number;
}

export interface MemoryResult {
  hbmBandwidthUtil: number;
  l2HitRate?: number;
  flashAttnSavings?: number;
}

export interface SchedulerResult {
  smOccupancy: number;
  warpStallRate?: number;
  activeWarps?: number;
}

export type BottleneckAlert =
  | { type: "memory_bound"; mfu: number; hbmUtil: number; tick: number }
  | { type: "low_occupancy"; smOccupancy: number; tick: number }
  | { type: "gradient_explosion"; gradNorm: number; tick: number }
  | { type: "loss_plateau"; recentLossDelta: number; tick: number };

export interface StartRunRequest {
  modelId: string;
  gpuId: string;
  source: DataSource;
  options?: {
    durationTicks?: number;
    tickMs?: number;
    seed?: number;
    flashAttention?: boolean;
    workflow?: SimulationWorkflow;
  };
}

export interface RunSummary {
  runId: string;
  modelId: string;
  gpuId: string;
  source: DataSource;
  startTimeMs: number;
  endTimeMs?: number;
  totalTicks: number;
}

export type LiveMessage =
  | { type: "snapshot"; runId: string; snapshot: MetricSnapshot }
  | { type: "alert"; runId: string; alert: BottleneckAlert };

/** Static edges describing how metrics influence each other across the simulated stack. */
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
