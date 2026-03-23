# Metrics Reference

This document defines every metric the simulation engine tracks, explains what it measures physically, states healthy target ranges, and provides a diagnosis guide for when values fall outside those ranges.

---

## Primary metrics

### MFU — Model FLOP Utilization

**Definition:** The fraction of the GPU's theoretical peak FLOP/s that the model actually uses during training.

```
MFU = (achieved FLOP/s) / (theoretical peak FLOP/s)
```

For an NVIDIA A100 80GB SXM4, theoretical peak in BF16 is 312 TFLOPS. If a training step achieves 156 TFLOPS, MFU = 0.5 (50%).

**Healthy range:** 40–60% for well-optimized transformer training. World-class runs (e.g., Chinchilla, PaLM) report 40–55%.

| Range | Interpretation |
|---|---|
| > 55% | Excellent — near best-in-class efficiency |
| 40–55% | Good — expected for optimized training |
| 25–40% | Marginal — investigate memory bottlenecks |
| < 25% | Poor — likely memory-bound or launch overhead issue |

**Research context:** Hoffmann et al. (2022) reported ~45% MFU for Chinchilla. Karpathy's nanoGPT achieves ~57% MFU on A100. PaLM reported ~46.2% MFU across 6144 TPUv4 chips.

> Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). **Training compute-optimal large language models (Chinchilla).** arXiv:2203.15556. https://arxiv.org/abs/2203.15556

---

### HBM bandwidth utilization

**Definition:** The fraction of the GPU's peak High Bandwidth Memory (HBM) read/write throughput that is actively used.

```
HBM util = (measured bytes/sec) / (peak bytes/sec)
```

A100 SXM4 peak HBM bandwidth: 2 TB/s. H100 SXM5: 3.35 TB/s.

**Healthy range:** 60–85% for memory-bound workloads (attention-heavy models). Compute-bound operations (GEMM on large matrices) may intentionally run at lower HBM utilization.

| Range | Interpretation |
|---|---|
| > 80% | Memory-bound — consider FlashAttention or quantization |
| 60–80% | Balanced — healthy for attention-heavy workloads |
| 30–60% | Compute-bound — GEMM is the bottleneck, not memory |
| < 30% | Underutilized — possible kernel launch inefficiency |

**Bottleneck signal:** If MFU is low AND HBM utilization is high, the model is memory-bound. The fix is usually FlashAttention, gradient checkpointing, or sequence length reduction.

---

### SM occupancy

**Definition:** The ratio of active warps to the maximum number of warps that could theoretically reside on a Streaming Multiprocessor (SM) simultaneously.

```
SM occupancy = (active warps per SM) / (max warps per SM)
```

NVIDIA H100 supports up to 64 warps per SM. If 38 are active, occupancy = 59%.

**Healthy range:** 50–80% is typical and sufficient. 100% occupancy is rarely achievable and not always beneficial.

| Range | Interpretation |
|---|---|
| > 65% | Good — latency hiding is effective |
| 40–65% | Acceptable — moderate stalls expected |
| 20–40% | Low — thread block config likely suboptimal |
| < 20% | Very low — check shared memory allocation and register pressure |

**Common causes of low occupancy:**
- Thread blocks are too small (fewer threads → fewer warps)
- Too much shared memory allocated per block (limits how many blocks fit per SM)
- High register pressure per thread (limits resident warps)

---

### Training loss

**Definition:** The scalar value of the loss function computed over the current mini-batch. For language models, typically cross-entropy loss over the vocabulary.

**What to watch:**
- Smooth monotonic decline across steps indicates healthy training
- Spikes indicate gradient instability or bad data batches
- Plateau for > 1000 steps may indicate learning rate is too low or the model has converged

**Research context:** Loss curves and their shapes are analyzed in depth in:

> Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2022). **Scaling vision transformers.** In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. arXiv:2106.04560. https://arxiv.org/abs/2106.04560

---

### Gradient norm

**Definition:** The L2 norm of all gradient tensors concatenated. A global measure of how large the parameter updates are.

```
grad_norm = sqrt(sum(p.grad ** 2) for p in model.parameters())
```

**Healthy range:** Typically 0.1–2.0 for well-tuned training. Most practitioners clip at 1.0.

| Value | Interpretation |
|---|---|
| < 0.05 | Vanishing gradients — network may not be learning |
| 0.1–1.0 | Healthy — normal update magnitude |
| 1.0–5.0 | Elevated — monitor for instability |
| > 10.0 | Gradient explosion — training will diverge without clipping |

**Gradient clipping:** The standard mitigation is `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`, which rescales gradients proportionally when the norm exceeds the threshold.

> Pascanu, R., Mikolov, T., & Bengio, Y. (2013). **On the difficulty of training recurrent neural networks.** In *Proceedings of the 30th International Conference on Machine Learning (ICML)*. arXiv:1211.5063. https://arxiv.org/abs/1211.5063

---

### Tokens per second (TPS)

**Definition:** The number of input tokens processed per second during training or the number of output tokens generated per second during inference.

**Training reference points (A100 80GB, BF16):**

| Model | TPS (training) |
|---|---|
| GPT-2 124M | ~180,000 |
| LLaMA 3 8B | ~14,000–18,000 |
| LLaMA 3 70B (8× A100) | ~3,500–5,000 |
| LLaMA 3.1 405B (64× H100) | ~800–1,200 |

**Inference reference points (single A100, BF16):**

| Model | TPS (generation) |
|---|---|
| LLaMA 3 8B | ~80–120 |
| LLaMA 3 70B | ~12–18 |

---

## Secondary metrics

### Warp stall rate

The fraction of warp execution cycles spent stalled (waiting for memory or data dependencies). High stall rates indicate the warp scheduler cannot hide latency effectively.

**Healthy target:** < 20% stall rate. Above 40% usually indicates a memory-bound kernel.

### L2 cache hit rate

The fraction of memory requests served from the L2 cache rather than HBM. Higher is better — cache hits are ~10× faster than HBM reads.

**Healthy target:** > 60% for attention kernels using FlashAttention tiling.

### Tensor core utilization

The fraction of tensor core cycles actively computing matrix multiplications vs idle. Separate from MFU — a kernel can have high MFU but low tensor core utilization if it spends cycles on non-GEMM operations.

**Healthy target:** > 70% for GEMM-heavy workloads.

---

## Bottleneck diagnosis matrix

Use this matrix when a training run underperforms:

| MFU | HBM util | SM occupancy | Diagnosis | Fix |
|---|---|---|---|---|
| Low | High | Medium | Memory-bound | Enable FlashAttention; reduce batch size |
| Low | Low | Low | Launch overhead / small kernels | Increase batch size; fuse kernels |
| Low | Medium | Low | Low occupancy | Increase threads/block; reduce shared mem |
| Medium | High | High | Normal memory-bound | Expected for attention; use FlashAttention-2 |
| High | High | High | Optimal | Nothing to fix |
| Low | Low | High | CPU bottleneck | Overlap data loading; use `pin_memory=True` |

---

## Profiling tools

| Tool | What it measures | Command |
|---|---|---|
| `nvidia-smi` | GPU utilization, VRAM, temperature | `nvidia-smi dmon -s u` |
| Nsight Systems | Full timeline: CPU, GPU, memory, kernels | `nsys profile python train.py` |
| Nsight Compute | Per-kernel metrics: occupancy, warp stalls, memory | `ncu --metrics sm__warps_active python train.py` |
| PyTorch Profiler | Operator-level timing, FLOP counts | See `DATA_SOURCES.md` |
| W&B | Training curves, system metrics over time | `wandb.log({"loss": loss})` |
