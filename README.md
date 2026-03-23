# GPU Algorithm Simulation Engine

A TypeScript-based simulation engine that orchestrates the full GPU algorithm stack — from model-level backpropagation down to silicon-level tensor core execution — fed by real public benchmark data.

**Repository:** [github.com/nikhilkanamadi/gpu-simulation-engine](https://github.com/nikhilkanamadi/gpu-simulation-engine)  
**Static docs (GitHub Pages):** publish the [`docs/`](docs/) folder from repository Settings → Pages.

---

## Project overview

This engine models the logical execution order of every major algorithm that runs inside a GPU during AI training and inference. It ingests public trace data from MLPerf, PyTorch Kineto, Meta HTA, and Weights & Biases, normalizes them into a unified `MetricSnapshot` interface, and replays them tick-by-tick through a modular algorithm pipeline.

The live dashboard surfaces MFU, HBM bandwidth utilization, SM occupancy, training loss, and bottleneck alerts in real time.

---

## Repository structure

```
gpu-sim-engine/
├── README.md                        ← this file
├── ARCHITECTURE.md                  ← system design and component map
├── ALGORITHMS.md                    ← algorithm stack with paper references
├── DATA_SOURCES.md                  ← public data ingestion guide
├── METRICS.md                       ← metric definitions and healthy ranges
├── src/
│   ├── ingestion/
│   │   ├── MLPerfParser.ts
│   │   ├── KinetoParser.ts
│   │   ├── WandbFetcher.ts
│   │   └── Normalizer.ts
│   ├── engine/
│   │   ├── Orchestrator.ts
│   │   ├── SimulationClock.ts
│   │   ├── modules/
│   │   │   ├── ModelModule.ts
│   │   │   ├── ComputeModule.ts
│   │   │   ├── MemoryModule.ts
│   │   │   └── WarpScheduler.ts
│   │   ├── MetricCollector.ts
│   │   └── BottleneckDetector.ts
│   ├── storage/
│   │   └── TimeSeriesStore.ts
│   ├── api/
│   │   └── server.ts
│   └── dashboard/
│       └── App.tsx
├── data/
│   ├── mlperf/
│   ├── kineto/
│   └── wandb/
└── tests/
```

---

## Quick start

```bash
# Install dependencies
npm install

# Run the simulation engine (development)
npm run dev

# Run with a specific model trace
npm run simulate -- --model llama-3.1-8b --gpu h100 --source mlperf

# Launch the dashboard
npm run dashboard
```

---

## Key concepts

| Concept | Description |
|---|---|
| `MetricSnapshot` | Normalized metric record emitted per simulation tick |
| `SimTick` | Clock event dispatched to each algorithm module |
| `AlgoResult` | Per-module output merged by `MetricCollector` |
| MFU | Model FLOP Utilization — primary efficiency signal |
| HBM bandwidth util | Memory subsystem health indicator |
| SM occupancy | Thread scheduler saturation metric |

---

## Research references

The algorithms simulated in this engine are each grounded in peer-reviewed, publicly available research. Full references are in [ALGORITHMS.md](./ALGORITHMS.md).

| Algorithm | Paper | Year |
|---|---|---|
| Backpropagation | Rumelhart et al., *Nature* | 1986 |
| Adam optimizer | Kingma & Ba, arXiv:1412.6980 | 2014 |
| Transformer / Attention | Vaswani et al., arXiv:1706.03762 | 2017 |
| CUDA / SIMT | Nickolls et al., *IEEE Micro* | 2008 |
| Tiled GEMM | Volkov & Demmel, SC08 | 2008 |
| Mixed precision training | Micikevicius et al., arXiv:1710.03740 | 2017 |
| FlashAttention | Dao et al., arXiv:2205.14135 | 2022 |
| FlashAttention-2 | Dao, arXiv:2307.08691 | 2023 |
| Warp scheduling | Narasiman et al., *MICRO* | 2011 |
| GPU memory model | Hong & Kim, *ISCA* | 2009 |

---

## License

MIT
