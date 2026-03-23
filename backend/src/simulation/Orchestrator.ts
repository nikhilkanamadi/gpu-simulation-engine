import type { MetricSnapshot, PipelineStageInfo } from "@gpu-sim/shared";

import { BottleneckDetector } from "./BottleneckDetector";
import { MetricCollector } from "./MetricCollector";
import { finalizePipelineTrace, snapshotSnippet } from "./pipelineTrace";
import type { AlgorithmModule, SimTick } from "@gpu-sim/shared";

import { ComputeModule } from "./modules/ComputeModule";
import { MemoryModule } from "./modules/MemoryModule";
import { ModelModule } from "./modules/ModelModule";
import { WarpScheduler } from "./modules/WarpScheduler";

export interface OrchestratorDeps {
  tickMs: number;
  durationTicks: number;
}

export class Orchestrator {
  private modules: AlgorithmModule<any>[] = [];
  private readonly collector: MetricCollector;
  private readonly detector: BottleneckDetector;
  private startEpochMs: number = Date.now();

  constructor(deps?: Partial<OrchestratorDeps>) {
    this.collector = new MetricCollector();
    this.detector = new BottleneckDetector();
    // Order is execution order: each stage reads the snapshot updated by prior stages.
    this.modules = [new ModelModule(), new ComputeModule(), new MemoryModule(), new WarpScheduler()];
  }

  reset() {
    this.startEpochMs = Date.now();
    this.detector.reset();
    for (const m of this.modules) m.reset();
  }

  processTick(baseSnapshot: MetricSnapshot, tick: number): { snapshot: MetricSnapshot; alert: any | null } {
    const clockMs = this.startEpochMs + tick;
    let working: MetricSnapshot = { ...baseSnapshot, tick };

    const model = this.modules.find((m) => m.name === "model") as AlgorithmModule<any>;
    const compute = this.modules.find((m) => m.name === "compute") as AlgorithmModule<any>;
    const memory = this.modules.find((m) => m.name === "memory") as AlgorithmModule<any>;
    const warp = this.modules.find((m) => m.name === "warp_scheduler") as AlgorithmModule<any>;

    const simTick = (snapshot: MetricSnapshot): SimTick => ({ tick, snapshot, clockMs });
    const stages: PipelineStageInfo[] = [];

    const modelRes = model.step(simTick(working));
    working = this.collector.applyModel(working, modelRes);
    stages.push({
      order: 1,
      moduleId: "model",
      title: "Model & gradients",
      summary: "Loss and gradient norms (replay or synthetic) set the optimization signal for this tick.",
      readsMetrics: [],
      producesMetrics: ["loss", "gradNorm"],
      valuesAtStage: snapshotSnippet(working, ["loss", "gradNorm"])
    });

    const computeRes = compute.step(simTick(working));
    working = this.collector.applyCompute(working, computeRes);
    stages.push({
      order: 2,
      moduleId: "compute",
      title: "Compute / GEMM",
      summary: "MFU and tensor-core utilization absorb gradient-heavy matmul work from the model stage.",
      readsMetrics: ["loss", "gradNorm"],
      producesMetrics: ["mfu", "tflopsAchieved", "tensorCoreUtil"],
      valuesAtStage: snapshotSnippet(working, ["mfu", "tflopsAchieved", "tensorCoreUtil"])
    });

    const memRes = memory.step(simTick(working));
    working = this.collector.applyMemory(working, memRes);
    stages.push({
      order: 3,
      moduleId: "memory",
      title: "Memory & caches",
      summary: "HBM traffic and L2 behavior follow compute intensity; flash-attn savings reflect IO-aware patterns.",
      readsMetrics: ["mfu", "tensorCoreUtil"],
      producesMetrics: ["hbmBandwidthUtil", "l2HitRate", "flashAttnSavings"],
      valuesAtStage: snapshotSnippet(working, ["hbmBandwidthUtil", "l2HitRate", "flashAttnSavings"])
    });

    const warpRes = warp.step(simTick(working));
    working = this.collector.applyWarp(working, warpRes);
    stages.push({
      order: 4,
      moduleId: "warp_scheduler",
      title: "Warp scheduler",
      summary: "SM occupancy and warp stalls close the loop between memory latency and runnable warps.",
      readsMetrics: ["hbmBandwidthUtil", "l2HitRate", "mfu"],
      producesMetrics: ["smOccupancy", "warpStallRate"],
      valuesAtStage: snapshotSnippet(working, ["smOccupancy", "warpStallRate"], {
        activeWarps: warpRes.activeWarps
      })
    });

    const out: MetricSnapshot = {
      ...working,
      pipelineTrace: finalizePipelineTrace(tick, stages)
    };

    const alert = this.detector.detect(out);
    return { snapshot: out, alert };
  }
}

