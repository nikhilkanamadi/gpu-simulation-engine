# Data Sources & Ingestion Guide

This document describes every public data source the simulation engine supports, how to access it, what schema it produces, and which models it covers.

All sources are free. No paid subscriptions are required.

---

## Overview

| Source | Format | Models covered | Update frequency |
|---|---|---|---|
| MLCommons MLPerf | JSON | LLaMA 3.1, BERT, ResNet, Stable Diffusion | Per benchmark round (~2× / year) |
| PyTorch Kineto | Chrome trace JSON | Any PyTorch model | On demand (self-generated) |
| Meta HTA | JSON bundles | Large-scale distributed training | Static sample traces |
| Weights & Biases | REST API (JSON) | GPT-2, LLaMA fine-tunes, diffusion models | Live (community runs) |
| Lambda Labs benchmarks | HTML table | LLaMA 2/3, Mistral, Stable Diffusion | Monthly |
| NVIDIA DL Perf Hub | CSV / web | GPT-3, Megatron-LM, T5, BERT | Per GPU release |

---

## Source 1 — MLCommons MLPerf

### About

MLPerf Training is the industry-standard AI benchmark suite. Results are peer-reviewed, publicly published, and cover every major GPU: NVIDIA H100/A100/Blackwell, AMD MI300X, Intel Gaudi, and Google TPU.

**URL:** https://mlcommons.org/benchmarks/training/

### Access

Results are published as structured JSON after each round. Download directly:

```bash
curl -O https://mlcommons.org/results/training-v4.1/results.json
```

### Relevant fields

```json
{
  "model": "llama2-70b",
  "system": "NVIDIA DGX H100",
  "num_accelerators": 8,
  "time_to_train_minutes": 47.3,
  "samples_per_second": 18420,
  "energy_per_epoch_joules": 142000
}
```

### Parser target: `MLPerfParser.ts`

Maps to `MetricSnapshot`:

- `time_to_train_minutes` → derive simulated `tokensPerSec`
- `samples_per_second` → `mfu` (estimate against theoretical peak)
- `system` → `gpuId`
- `model` → `modelId`

### Models available

- LLaMA 2 70B, LLaMA 3.1 405B
- BERT-large
- ResNet-50
- Stable Diffusion XL
- Flux.1
- GPT-J 6B

---

## Source 2 — PyTorch Kineto traces

### About

PyTorch's built-in profiler captures every CUDA kernel invocation, memory allocation, and CPU↔GPU synchronization event during a training step and exports them as Chrome DevTools trace JSON. You generate these yourself for any model.

**Documentation:** https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

### Generating a trace

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = ...  # any PyTorch model

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    with record_function("model_inference"):
        output = model(input)

prof.export_chrome_trace("trace.json")
```

### Relevant fields in trace.json

```json
{
  "traceEvents": [
    {
      "name": "volta_sgemm_128x32_sliced1x4_tn",
      "cat": "cuda_runtime",
      "ph": "X",
      "dur": 412,
      "args": {
        "External id": 42,
        "device": 0,
        "stream": 7,
        "grid": [128, 1, 1],
        "block": [32, 1, 1],
        "flops": 536870912
      }
    }
  ]
}
```

### Parser target: `KinetoParser.ts`

Maps to `KernelTrace[]` then to `MetricSnapshot`:

- `dur` (microseconds) → kernel duration timeline
- `flops` → achieved TFLOPS calculation
- `grid` / `block` → SM occupancy estimate
- Memory events → HBM bandwidth utilization

### Free trace datasets

Pre-generated traces for common models are available in the HTA repository (see Source 3) and on HuggingFace datasets:

```
huggingface.co/datasets/pytorch/kineto-traces
```

---

## Source 3 — Meta Holistic Trace Analysis (HTA)

### About

HTA is Meta's open-source library for analyzing distributed training performance at scale. The repository ships with sample trace bundles from real large-scale training jobs. It surfaces metrics that raw Kineto traces don't expose directly, such as GPU idle time breakdown and communication/computation overlap.

**Repository:** https://github.com/facebookresearch/HolisticTraceAnalysis

### Access

```bash
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git
# Sample traces located at:
# HolisticTraceAnalysis/tests/data/
```

### Relevant metrics exposed

| Metric | Description |
|---|---|
| `idle_time_pct` | Fraction of time GPU is idle waiting for CPU or data |
| `comm_compute_overlap_pct` | How much communication overlaps with compute |
| `kernel_duration_p50_us` | Median kernel execution time in microseconds |
| `queue_length_avg` | Average CUDA kernel queue depth |

### Parser target: `HTAParser.ts`

Maps primarily to `MemResult` and `SchedResult` fields in `MetricSnapshot`.

---

## Source 4 — Weights & Biases public runs

### About

Thousands of researchers share their training runs publicly on the W&B platform. The public API exposes full metric histories — loss, gradient norm, GPU utilization, learning rate, throughput — step by step.

**API docs:** https://docs.wandb.ai/ref/python/public-api/api

### Access

```typescript
// WandbFetcher.ts — no auth needed for public runs
const BASE = "https://api.wandb.ai/api/v1";

async function fetchRunHistory(entity: string, project: string, runId: string) {
  const res = await fetch(
    `${BASE}/runs/${entity}/${project}/${runId}/history?samples=1000`
  );
  return res.json();
}
```

### Example public runs

| Model | Entity/Project | Run contains |
|---|---|---|
| GPT-2 medium | `openai/gpt-2` | loss, grad_norm, lr |
| LLaMA fine-tune | `huggingface/llama-finetune` | loss, gpu_util, tokens/sec |
| Stable Diffusion | `stability/sdxl` | loss, fid_score, vram_used |

### Relevant fields per history row

```json
{
  "_step": 1500,
  "train/loss": 2.14,
  "train/grad_norm": 0.83,
  "train/lr": 0.00029,
  "system/gpu.0.gpu": 94.2,
  "system/gpu.0.memoryAllocated": 38.7,
  "perf/tokens_per_sec": 14200
}
```

### Parser target: `WandbFetcher.ts`

Maps `_step` → `tick`, loss fields → `loss` / `gradNorm`, system fields → `smOccupancy` estimate, `tokens_per_sec` → `tokensPerSec`.

---

## Source 5 — Lambda Labs GPU benchmarks

### About

Lambda Labs publishes regularly updated GPU benchmark tables showing training throughput for popular models across GPU SKUs. Useful for anchoring simulation outputs against real-world baselines.

**URL:** https://lambdalabs.com/gpu-benchmarks

### Access

The page renders as an HTML table. A simple scraper extracts the CSV:

```typescript
// LambdaScraper.ts
import * as cheerio from "cheerio";

async function scrapeLambdaBenchmarks(): Promise<BenchmarkRow[]> {
  const html = await fetch("https://lambdalabs.com/gpu-benchmarks").then(r => r.text());
  const $ = cheerio.load(html);
  // parse table rows → BenchmarkRow[]
}
```

### Schema

```typescript
interface BenchmarkRow {
  model: string;        // "LLaMA 3 8B"
  gpuSku: string;       // "H100 SXM5 80GB"
  tokensPerSec: number;
  batchSize: number;
  precision: "fp16" | "bf16" | "fp32" | "int8";
}
```

---

## Adding a new source

To add a new data source:

1. Create `src/ingestion/YourSourceParser.ts`
2. Implement the `DataParser<T>` interface:

```typescript
interface DataParser<TRaw> {
  name: DataSource;
  parse(raw: TRaw): MetricSnapshot[];
  validate(raw: unknown): raw is TRaw;
}
```

3. Register it in `src/ingestion/Normalizer.ts`
4. Add a test in `tests/ingestion/YourSourceParser.test.ts`

No other files need to change. The normalization boundary ensures the engine core is completely unaware of the new source.

---

## Data directory layout

```
data/
├── mlperf/
│   ├── results-v4.1.json
│   └── results-v4.0.json
├── kineto/
│   ├── gpt2-medium-a100.json
│   ├── resnet50-h100.json
│   └── llama3-8b-h100.json
├── hta/
│   └── sample-traces/
│       ├── trace_rank0.json
│       └── trace_rank1.json
└── wandb/
    └── cached-runs/
        └── llama-finetune-run42.json
```
