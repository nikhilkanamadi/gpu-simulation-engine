import type { ComputeResult, MemoryResult, ModelResult, MetricSnapshot, SchedulerResult } from "@gpu-sim/shared";

export class MetricCollector {
  applyModel(base: MetricSnapshot, model: ModelResult): MetricSnapshot {
    return {
      ...base,
      loss: model.loss ?? base.loss,
      gradNorm: model.gradNorm ?? base.gradNorm
    };
  }

  applyCompute(base: MetricSnapshot, compute: ComputeResult): MetricSnapshot {
    return {
      ...base,
      mfu: compute.mfu ?? base.mfu,
      tflopsAchieved: compute.tflopsAchieved ?? base.tflopsAchieved,
      tensorCoreUtil: compute.tensorCoreUtil ?? base.tensorCoreUtil
    };
  }

  applyMemory(base: MetricSnapshot, memory: MemoryResult): MetricSnapshot {
    return {
      ...base,
      hbmBandwidthUtil: memory.hbmBandwidthUtil ?? base.hbmBandwidthUtil,
      l2HitRate: memory.l2HitRate ?? base.l2HitRate,
      flashAttnSavings: memory.flashAttnSavings ?? base.flashAttnSavings
    };
  }

  applyWarp(base: MetricSnapshot, warp: SchedulerResult): MetricSnapshot {
    return {
      ...base,
      smOccupancy: warp.smOccupancy ?? base.smOccupancy,
      warpStallRate: warp.warpStallRate ?? base.warpStallRate,
      tokensPerSec: base.tokensPerSec ?? undefined
    };
  }

  collect(
    base: MetricSnapshot,
    parts: {
      model: ModelResult;
      compute: ComputeResult;
      memory: MemoryResult;
      warp: SchedulerResult;
    }
  ): MetricSnapshot {
    let s = this.applyModel(base, parts.model);
    s = this.applyCompute(s, parts.compute);
    s = this.applyMemory(s, parts.memory);
    s = this.applyWarp(s, parts.warp);
    return s;
  }
}

