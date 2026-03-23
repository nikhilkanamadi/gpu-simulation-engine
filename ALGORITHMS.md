# Algorithm Stack & Research References

This document maps every algorithm simulated in the engine to its founding research paper, describes what the algorithm does, and explains how the simulation module models its behavior.

All papers listed are freely available via arXiv, IEEE Xplore, or ACM Digital Library.

---

## Execution order

The engine runs algorithms in the following logical order on every simulation tick. Each layer depends on the one below it.

```
1. Model algorithm       (backpropagation, Adam, attention)
2. Parallelism planner   (CUDA / SIMT decomposition)
3. Linear algebra engine (GEMM, tiled matrix multiply)
4. Memory subsystem      (FlashAttention, coalescing, prefetch)
5. Thread scheduler      (warp dispatch, occupancy)
6. Silicon execution     (tensor cores, FP16 MACs)
```

---

## Layer 1 — Model algorithms

### Backpropagation

**What it does:** Computes gradients of the loss function with respect to every parameter in the network using the chain rule of calculus. Enables the network to learn by propagating error signals backward through layers.

**Simulation model:** The `ModelModule` tracks a simulated loss value and gradient norm per tick. When real W&B training curves are loaded, the module replays actual loss trajectories. Without real data, it uses a synthetic exponential decay with Gaussian noise.

**Research reference:**

> Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). **Learning representations by back-propagating errors.** *Nature*, 323(6088), 533–536. https://doi.org/10.1038/323533a0

---

### Adam optimizer

**What it does:** An adaptive gradient descent algorithm that maintains per-parameter first and second moment estimates of gradients. Automatically adjusts the effective learning rate for each parameter, making training far more stable than vanilla SGD.

**Simulation model:** The module tracks the effective learning rate schedule, the gradient norm trend, and detects loss plateaus indicative of optimizer stagnation.

**Key hyperparameters:** `lr = 3e-4`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`

**Research reference:**

> Kingma, D. P., & Ba, J. (2014). **Adam: A method for stochastic optimization.** arXiv preprint arXiv:1412.6980. https://arxiv.org/abs/1412.6980

---

### Transformer attention

**What it does:** Computes scaled dot-product attention between query, key, and value matrices across all heads in parallel. The core operation of every modern large language model.

**Simulation model:** The `ModelModule` models attention as a GEMM-equivalent operation with quadratic complexity in sequence length. Sequence length is a configurable parameter; longer sequences raise memory pressure on the `MemoryModule`.

**Research reference:**

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is all you need.** *Advances in Neural Information Processing Systems*, 30. arXiv:1706.03762. https://arxiv.org/abs/1706.03762

---

## Layer 2 — Parallelism planning

### CUDA / SIMT execution model

**What it does:** Translates a model's operations into thousands of parallel threads that execute the same instruction on different data simultaneously (Single Instruction, Multiple Threads). The CUDA programming model is what makes GPUs usable for general compute.

**Simulation model:** The `ComputeModule` models kernel launch overhead, thread block sizing, and the relationship between block configuration and SM occupancy.

**Research reference:**

> Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008). **Scalable parallel programming with CUDA.** *Queue*, 6(2), 40–53. https://doi.org/10.1145/1365490.1365500

---

## Layer 3 — Linear algebra engine

### Tiled GEMM (General Matrix-Matrix Multiplication)

**What it does:** The dominant compute operation in all AI workloads. Every fully connected layer, attention computation, and convolution reduces to GEMM. Tiling breaks large matrices into cache-sized submatrices so tensor cores are never starved for data.

**Simulation model:** The `ComputeModule` tracks theoretical TFLOPS vs achieved TFLOPS. The ratio is the MFU (Model FLOP Utilization). Tile size is configurable; smaller tiles simulate cache misses and lower MFU.

**Research reference:**

> Volkov, V., & Demmel, J. W. (2008). **Benchmarking GPUs to tune dense linear algebra.** In *Proceedings of the 2008 ACM/IEEE Conference on Supercomputing (SC08)*. https://doi.org/10.1109/SC.2008.5213359

---

### Mixed precision training (FP16 / BF16)

**What it does:** Trains neural networks in half-precision (FP16 or BF16) instead of full single precision (FP32), roughly halving memory usage and doubling tensor core throughput with no meaningful accuracy loss when loss scaling is applied.

**Simulation model:** The `ComputeModule` applies a 1.4× MFU multiplier when FP16 precision is selected, reflecting the throughput advantage of tensor cores over standard CUDA cores.

**Research reference:**

> Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Wu, H. (2017). **Mixed precision training.** arXiv preprint arXiv:1710.03740. https://arxiv.org/abs/1710.03740

---

## Layer 4 — Memory subsystem

### FlashAttention

**What it does:** A reordering of the attention computation that tiles the operation to fit in SRAM (the GPU's fast on-chip memory), dramatically reducing reads from slow HBM. Makes long-context inference (100K+ tokens) practically feasible.

**Simulation model:** The `MemoryModule` models HBM read reduction as a function of sequence length. When FlashAttention is enabled (default), HBM bandwidth utilization drops by a configurable factor relative to naive attention.

**Research references:**

> Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). **FlashAttention: Fast and memory-efficient exact attention with IO-awareness.** *Advances in Neural Information Processing Systems*, 35. arXiv:2205.14135. https://arxiv.org/abs/2205.14135

> Dao, T. (2023). **FlashAttention-2: Faster attention with better parallelism and work partitioning.** arXiv:2307.08691. https://arxiv.org/abs/2307.08691

---

### GPU memory access and coalescing

**What it does:** Thread memory access patterns must be aligned so the GPU can fetch 128-byte cache lines efficiently. Coalesced access means all threads in a warp access contiguous memory addresses, maximizing HBM bandwidth.

**Simulation model:** The `MemoryModule` models bandwidth efficiency as a function of access alignment. Misaligned access patterns (non-coalesced) reduce effective bandwidth by a penalty factor.

**Research reference:**

> Hong, S., & Kim, H. (2009). **An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness.** In *Proceedings of the 36th Annual International Symposium on Computer Architecture (ISCA '09)*. https://doi.org/10.1145/1555754.1555775

---

## Layer 5 — Thread scheduler

### Warp scheduling and latency hiding

**What it does:** GPUs hide memory latency by swapping in new warps (groups of 32 threads) whenever the current warp stalls waiting for data. SM occupancy — the fraction of maximum resident warps that are active — is the key metric. High occupancy allows more latency hiding.

**Simulation model:** The `WarpScheduler` module tracks active warps per SM, warp stall rate, and occupancy. Low occupancy triggers a bottleneck alert.

**Research reference:**

> Narasiman, V., Shebanow, M., Lee, C. J., Miftakhutdinov, R., Mutlu, O., & Patt, Y. N. (2011). **Improving GPU performance via large warps and two-level warp scheduling.** In *Proceedings of the 44th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-44)*. https://doi.org/10.1145/2155620.2155656

---

## Summary table

| Layer | Algorithm | Paper | Year | Free access |
|---|---|---|---|---|
| Model | Backpropagation | Rumelhart et al., *Nature* | 1986 | [nature.com](https://www.nature.com/articles/323533a0) |
| Model | Adam optimizer | Kingma & Ba | 2014 | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| Model | Transformer attention | Vaswani et al. | 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| Parallelism | CUDA / SIMT | Nickolls et al., *Queue* | 2008 | [ACM DL](https://dl.acm.org/doi/10.1145/1365490.1365500) |
| Compute | Tiled GEMM | Volkov & Demmel, SC08 | 2008 | [IEEE](https://ieeexplore.ieee.org/document/5213359) |
| Compute | Mixed precision | Micikevicius et al. | 2017 | [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) |
| Memory | FlashAttention | Dao et al. | 2022 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| Memory | FlashAttention-2 | Dao | 2023 | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) |
| Memory | GPU memory model | Hong & Kim, *ISCA* | 2009 | [ACM DL](https://dl.acm.org/doi/10.1145/1555754.1555775) |
| Scheduler | Warp scheduling | Narasiman et al., *MICRO* | 2011 | [ACM DL](https://dl.acm.org/doi/10.1145/2155620.2155656) |
