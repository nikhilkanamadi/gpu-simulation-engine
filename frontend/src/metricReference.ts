export interface MetricReferenceItem {
  key: string;
  name: string;
  definition: string;
  healthy: string;
  interpretation: string;
}

export const METRIC_REFERENCE: MetricReferenceItem[] = [
  {
    key: "mfu",
    name: "MFU (Model FLOP Utilization)",
    definition:
      "Fraction of theoretical peak FLOP/s that the model achieves during training.",
    healthy: "40-60% for optimized transformer training.",
    interpretation:
      "Low MFU means poor compute efficiency; high MFU indicates strong hardware utilization."
  },
  {
    key: "hbmBandwidthUtil",
    name: "HBM Bandwidth Utilization",
    definition:
      "Fraction of peak HBM memory throughput currently used by workload reads/writes.",
    healthy: "60-85% for memory-bound attention-heavy workloads.",
    interpretation:
      "High HBM + low MFU usually signals memory bottlenecks; lower HBM can indicate compute-bound kernels."
  },
  {
    key: "smOccupancy",
    name: "SM Occupancy",
    definition:
      "Ratio of active warps per Streaming Multiprocessor to maximum resident warps.",
    healthy: "50-80% is typically effective.",
    interpretation:
      "Low occupancy reduces latency hiding and can hurt throughput."
  },
  {
    key: "trainingLoss",
    name: "Training Loss",
    definition: "Scalar loss value over the current mini-batch (often cross-entropy for LLMs).",
    healthy: "Should trend downward smoothly over time.",
    interpretation:
      "Spikes suggest instability; long plateaus can indicate convergence or suboptimal learning rate."
  },
  {
    key: "gradientNorm",
    name: "Gradient Norm",
    definition: "Global L2 norm across gradients of all trainable parameters.",
    healthy: "Typically ~0.1-2.0; clipping often near 1.0.",
    interpretation:
      "Very high values suggest exploding gradients; very low values can indicate vanishing gradients."
  },
  {
    key: "tokensPerSec",
    name: "Tokens per Second (TPS)",
    definition:
      "Tokens processed per second during training, or tokens generated per second during inference.",
    healthy: "Model and hardware dependent; should be stable and improve with optimization.",
    interpretation:
      "Throughput KPI for end-to-end system performance."
  },
  {
    key: "warpStallRate",
    name: "Warp Stall Rate",
    definition: "Fraction of warp cycles spent stalled waiting on dependencies or memory.",
    healthy: "Below 20% is a good target.",
    interpretation:
      "High stall rate indicates scheduler cannot hide latency effectively."
  },
  {
    key: "l2CacheHitRate",
    name: "L2 Cache Hit Rate",
    definition: "Fraction of memory requests served from L2 cache instead of HBM.",
    healthy: "Above 60% for attention kernels with effective tiling.",
    interpretation:
      "Higher hit rate reduces expensive HBM traffic and improves memory efficiency."
  },
  {
    key: "tensorCoreUtil",
    name: "Tensor Core Utilization",
    definition:
      "Fraction of tensor core compute cycles actively performing matrix operations.",
    healthy: "Above 70% for GEMM-heavy workloads.",
    interpretation:
      "Low tensor core utilization indicates lost acceleration opportunities in compute kernels."
  },
  {
    key: "kvCacheUtil",
    name: "KV Cache Utilization",
    definition:
      "Fraction of allocated key/value cache capacity currently used by active context and generation steps.",
    healthy: "Inference-heavy runs should stay high but below saturation (~40-85%).",
    interpretation:
      "High values indicate long-context pressure; near saturation can trigger paging or latency spikes."
  },
  {
    key: "energyWatts",
    name: "GPU Energy Draw (Watts)",
    definition:
      "Approximate GPU board power draw during the current simulation tick.",
    healthy: "Depends on SKU and workflow; sustained peak draw is expected during throughput benchmarks.",
    interpretation:
      "Use with throughput metrics to estimate efficiency (tokens/sec per watt)."
  },
  {
    key: "cpuInputUtil",
    name: "CPU Input Pipeline Utilization",
    definition:
      "Estimated CPU-side utilization for data loading, preprocessing, and input staging.",
    healthy: "50-80% in balanced training pipelines.",
    interpretation:
      "High CPU input utilization with low MFU can indicate host-side bottlenecks starving the GPU."
  }
];

