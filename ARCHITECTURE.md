# System Architecture

This document describes the full system design of the GPU Algorithm Simulation Engine: its five architectural zones, the interfaces between them, and the rationale behind key design decisions.

---

## Architectural zones

The system is organized into five vertical layers, each with a single responsibility.

```
┌─────────────────────────────────────────────────────────┐
│               External data sources                     │
│   MLPerf · Kineto traces · W&B API · HTA · Lambda       │
└────────────────────────┬────────────────────────────────┘
                         │ raw JSON / CSV / trace files
┌────────────────────────▼────────────────────────────────┐
│           Ingestion & normalization layer                │
│   Parsers → schema validation → MetricSnapshot queue    │
└────────────────────────┬────────────────────────────────┘
                         │ MetricSnapshot[]
┌────────────────────────▼────────────────────────────────┐
│              Core simulation engine                     │
│  Orchestrator → clock → modules → MetricCollector       │
└──────────┬─────────────────────────────────┬────────────┘
           │ INSERT rows                     │ ws.send()
┌──────────▼──────────┐         ┌────────────▼────────────┐
│  Time-series store  │         │    API server           │
│  SQLite / TimescaleDB│        │    Express + WebSocket   │
└─────────────────────┘         └────────────┬────────────┘
                                             │ live stream
                                ┌────────────▼────────────┐
                                │   React dashboard       │
                                │   Recharts · gauges     │
                                └─────────────────────────┘
```

---

## Zone 1 — External data sources

Five public sources feed raw data into the engine. Each is accessed differently:

| Source | Access method | Format |
|---|---|---|
| MLCommons MLPerf | Direct download from mlcommons.org/results | JSON |
| PyTorch Kineto | `prof.export_chrome_trace()` from any model | Chrome trace JSON |
| Meta HTA | GitHub repo — sample traces bundled | JSON |
| Weights & Biases | Public REST API `api.wandb.ai` | JSON (paginated) |
| Lambda Labs benchmarks | HTML table scrape from lambda.ai/gpu-benchmarks | HTML → CSV |

All sources are free and require no authentication except W&B (free API key).

---

## Zone 2 — Ingestion & normalization layer

### Design principle

Every upstream source speaks a different schema. The normalization boundary converts all of them into a single `MetricSnapshot` interface before any downstream component ever sees the data. This decouples the engine from source schema changes.

### MetricSnapshot interface

```typescript
interface MetricSnapshot {
  tick: number;
  modelId: string;              // "llama-3.1-8b", "resnet-50"
  gpuId: string;                // "h100-80gb", "a100-40gb"
  mfu: number;                  // 0–1, model FLOP utilization
  hbmBandwidthUtil: number;     // 0–1
  smOccupancy: number;          // 0–1
  tflopsAchieved?: number;
  loss?: number;
  gradNorm?: number;
  tokensPerSec?: number;
  warpStallRate?: number;
  source: DataSource;
  timestampMs: number;
}

type DataSource = "mlperf" | "kineto" | "hta" | "wandb" | "lambda" | "synthetic";
```

### Parser modules

Each parser is a pure function: raw input in, `MetricSnapshot[]` out.

```
MLPerfParser.ts     — parse results JSON, extract throughput + energy fields
KinetoParser.ts     — parse chrome trace, compute kernel durations + FLOP counts
HTAParser.ts        — parse HTA bundle, extract idle time + comm/compute overlap
WandbFetcher.ts     — call REST API, page through run history, extract loss + gpu_util
LambdaScraper.ts    — parse HTML table, extract tokens/sec per GPU model
```

---

## Zone 3 — Core simulation engine

This is the heart of the system. It drives algorithm execution in the logical order established by the GPU compute pipeline.

### Orchestrator

The orchestrator owns the simulation clock and the module dispatch loop. On each tick it:

1. Dequeues the next `MetricSnapshot` from the ingestion queue
2. Constructs a `SimTick` event
3. Dispatches it to each algorithm module in order
4. Passes all module results to the `MetricCollector`
5. Emits the aggregated snapshot to storage and the WebSocket server

```typescript
interface SimTick {
  tick: number;
  snapshot: MetricSnapshot;
  clockMs: number;
}
```

### Algorithm modules

Each module implements the `AlgorithmModule` interface:

```typescript
interface AlgorithmModule<TResult> {
  name: string;
  step(tick: SimTick): TResult;
  reset(): void;
}
```

The four modules and their return types:

| Module | Return type | Key fields |
|---|---|---|
| `ModelModule` | `AlgoResult` | `loss`, `gradNorm`, `convergenceRate` |
| `ComputeModule` | `ComputeResult` | `tflopsAchieved`, `mfu`, `tensorCoreUtil` |
| `MemoryModule` | `MemResult` | `hbmBandwidthUtil`, `l2HitRate`, `flashAttnSavings` |
| `WarpScheduler` | `SchedResult` | `smOccupancy`, `warpStallRate`, `activeWarps` |

### MetricCollector

Merges all four module results into a single `MetricSnapshot` per tick and pushes it to two sinks: the time-series store and the WebSocket broadcast channel.

### BottleneckDetector

Runs inline on every emitted snapshot and fires typed alerts:

```typescript
type BottleneckAlert =
  | { type: "memory_bound";   mfu: number; hbmUtil: number }
  | { type: "low_occupancy";  smOccupancy: number }
  | { type: "gradient_explosion"; gradNorm: number }
  | { type: "loss_plateau";   recentLossDelta: number };
```

---

## Zone 4 — Storage

### Time-series store

Metric snapshots are written to a time-series table optimized for range queries (replay a specific run, compare two models over the same tick window).

```sql
CREATE TABLE metric_snapshots (
  tick          INTEGER,
  model_id      TEXT,
  gpu_id        TEXT,
  mfu           REAL,
  hbm_bw_util   REAL,
  sm_occupancy  REAL,
  loss          REAL,
  grad_norm     REAL,
  source        TEXT,
  recorded_at   INTEGER  -- Unix ms
);
CREATE INDEX idx_model_tick ON metric_snapshots (model_id, tick);
```

Use SQLite for local development, TimescaleDB for production deployments with large trace datasets.

### Run cache

Parsed trace files are expensive to re-parse. A simple file-based cache keyed on `sha256(rawFile)` avoids redundant work. Redis can replace the filesystem cache for multi-process deployments.

---

## Zone 5 — API server and dashboard

### API server (Express + WebSocket)

```
GET  /api/runs                    → list all ingested model runs
GET  /api/runs/:modelId/snapshots → paginated metric history
GET  /api/runs/:modelId/summary   → aggregated stats (avg MFU, peak loss, etc.)
WS   /ws/live                     → real-time MetricSnapshot stream
```

### React dashboard

Component tree:

```
App
├── RunSelector          — pick model + GPU from ingested runs
├── LiveMetricGrid
│   ├── MFUGauge
│   ├── HBMBandwidthBar
│   ├── SMOccupancyBar
│   └── LossDisplay
├── LossCurveChart       — Recharts LineChart, tick on x-axis
├── BottleneckAlertFeed  — live alert log with severity badges
└── SimulationControls   — play / pause / step / speed / reset
```

---

## Key design decisions

**Single normalization boundary.** All parsers produce `MetricSnapshot` and nothing else. No module ever imports a parser directly. This means adding a new data source requires only one new parser file.

**Modules are pure functions.** Each `step()` call takes a `SimTick` and returns a result with no side effects. This makes modules trivially unit-testable and replaceable.

**Clock drives everything.** The simulation clock is the single source of time. Replay speed is controlled by the tick interval, not by throttling data fetch.

**Bottleneck detection is synchronous.** The detector runs on the same tick as emission, so alerts appear in the dashboard at the same frame as the metric that triggered them.
