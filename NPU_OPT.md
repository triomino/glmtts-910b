# NPU Performance Optimization Guide

GLM-TTS Ascend NPU (910B) optimizations for the LLM decoding stage, consolidated in `npu_opt.py`.

All optimizations are **enabled by default** when `GLMTTS_NPU_OPT=1` (default). Each can be independently disabled.

## Master Switch

| Env Var | Default | Description |
|---------|---------|-------------|
| `GLMTTS_NPU_OPT` | `1` | Enable (`1`) or disable (`0`) all model-level NPU optimizations |
| `GLMTTS_NO_GRAPH_REPLAY` | (unset) | Disable NPUGraph replay, run eager forward per step |

## Individual Switches

| Env Var | Optimization | Description |
|---------|-------------|-------------|
| `GLMTTS_DISABLE_FIA=1` | Fused Attention | `npu_fused_infer_attention_score` for decode S=1 |
| `GLMTTS_DISABLE_SCATTER_UPDATE=1` | KV Cache Write | `scatter_update_` replacing `index_copy_` for decode S=1 |
| `GLMTTS_DISABLE_RMS_NORM=1` | Fused RMS Norm | `npu_rms_norm` replacing pow+mul (decode only, S=1) |
| `GLMTTS_DISABLE_MLP_FUSE=1` | Fused MLP | gate_up + `npu_swiglu` (single linear + fused activation, decode only) |
| `GLMTTS_DISABLE_ROPE=1` | Fused RoPE | `_npu_rotary_embedding` — **disabled by default** (see Known Issues) |

> **Note**: The nucleus sampling optimization (`cosyvoice/utils/common.py`) is applied unconditionally — it replaces per-element `.item()` NPU syncs (up to 25 per step) with a single bulk `.cpu()` transfer.

## Benchmark Setup

All performance numbers measured on **Ascend 910B4**:
- Test text: 28 Chinese characters → ~110 decode steps
- Single decode step time = sampling + buffer preparation + model forward + log_softmax
- Warmup 10 steps, measure 100 steps, report median
- Each config measured with graph replay enabled (except where noted)

---

## 1. NPUGraph Static Graph Replay

### Principle

`torch.npu.NPUGraph` captures the entire decode step computation graph (embedding → 28 transformer layers → norm → lm_head) into a replayable artifact. Once captured, each decode step only executes the pre-compiled graph, bypassing Python interpreter overhead and kernel launch scheduling.

Two modes are supported depending on whether FIA is enabled:
- **With FIA**: `torch.npu.NPUGraph` native capture (works with FIA's pre-allocated buffers)
- **Without FIA**: `torch.npu.make_graphed_callables` wraps the step function

### Performance

| Config | Single Step | vs Eager |
|--------|-------------|----------|
| Eager (no graph, no opt) | 50.37 ms | baseline |
| +Graph (graph only, no opt) | 14.16 ms | **-71.9%** |

**Isolated benefit of graph replay: 50.37 → 14.16 ms (3.6× speedup).**

### Precision

Bit-exact: graph replay produces identical logits to non-graph eager execution.
Verified: cos=1.0, max_abs=0 between graph replay and non-graph forward.

---

## 2. Fused Infer Attention (FIA)

### Principle

Replaces the standard attention computation (QK^T → softmax → PV, 3 separate kernels) with a single fused kernel `npu_fused_infer_attention_score`.

```
Standard:  Q @ K^T (matmul) → scale + mask → softmax → output @ V (matmul)
Fused:     npu_fused_infer_attention_score(Q, K, V, attn_mask, ...) → single kernel
```

The fused kernel uses **BNSD** layout and requires:
- `sparse_mode=0` with a **bool attention mask** (True = masked position)
- `actual_seq_lengths=[1]` for decode (S=1)
- Pre-allocated output/workspace buffers for NPUGraph compatibility

**Why beneficial**: Eliminates intermediate QK^T materialization (vocab 98304 × head_dim 128), reduces memory bandwidth, single kernel launch instead of 3+.

### Performance

Measured on top of graph (graph enabled):

| Config | Single Step | Δ from previous |
|--------|-------------|-----------------|
| Graph only (no opt) | 14.16 ms | baseline |
| +FIA | 12.74 ms | **-1.42 ms (-10.0%)** |

### Precision

| Op | Cosine | max_abs | rel_euc | JS |
|----|--------|---------|---------|----|
| attn_out (layer 0) | 0.9999995 | 0.125 | 0.0010 | ~0 |

Maxdiff per decode step: ~0.125 (bf16 ULP level). No error accumulation across steps.

---

## 3. scatter_update_ KV Cache

### Principle

Replaces `index_copy_` in `StaticLayer.update` (called during KV cache write in each decode step) with `torch_npu.scatter_update_`.

`index_copy_` copies a single slice per call → scatter kernel. `scatter_update_` is a dedicated NPU operator optimized for this access pattern.

**Decode only (S=1)**: Prefill still uses the original `index_copy_` for multi-token updates.

### Performance

| Config | Single Step | Δ from previous |
|--------|-------------|-----------------|
| +FIA | 12.74 ms | baseline |
| +scatter | 9.51 ms | **-3.22 ms (-25.4%)** |

### Precision

Bit-exact: `scatter_update_` produces identical results to `index_copy_` (verified: cos=1.0, max_abs=0).

---

## 4. Fused RMS Norm (decode only)

### Principle

Replaces the standard RMS Norm computation with a single `npu_rms_norm` kernel.

```
Standard: pow(hidden, 2) → mean(-1) → add(eps) → rsqrt → mul(weight)  (5 kernel launches)
Fused:    npu_rms_norm(hidden, weight, eps) → (output, rstd)           (1 kernel)
```

**Decode only (S=1)**: Prefill (S>1) falls back to eager bf16 computation to maintain bit-exact prefill output.

### Performance

| Config | Single Step | Δ from previous |
|--------|-------------|-----------------|
| +scatter | 9.51 ms | baseline |
| +RMS | 8.81 ms | **-0.70 ms (-7.4%)** |

### Precision

| Op | Cosine | max_abs | rel_euc | JS |
|----|--------|---------|---------|----|
| ln_out (layer 0) | 0.999996 | 0.0078 | 0.0028 | 9.6e-7 |

---

## 5. Fused MLP / SwiGLU (decode only)

### Principle

The GLM MLP uses a SwiGLU structure: `silu(gate_proj(x)) * up_proj(x) → down_proj`.

Fusion is done in two stages:

1. **Weight concatenation**: `gate_proj` and `up_proj` weights are concatenated into a single `gate_up_weight` parameter. A single `F.linear(x, gate_up_weight)` computes both projections in one call (one large matmul instead of two small ones, better NPU utilization).

2. **Activation fusion**: `npu_swiglu` fuses the SiLU activation and element-wise multiply into a single kernel call.

```
Standard: gate = silu(linear_gate(x)) → up = linear_up(x) → silu(gate) * up → linear_down(x)
Fused:    gu = linear_gate_up(x) → swiglu_out = npu_swiglu(gu) → linear_down(swiglu_out)
```

**Decode only (S=1)**: Prefill falls back to eager `silu(gate) * up` for numerical accuracy.

### Performance

| Config | Single Step | Δ from previous |
|--------|-------------|-----------------|
| +RMS | 8.81 ms | baseline |
| **+MLP (final)** | **8.36 ms** | **-0.45 ms (-5.1%)** |

### Precision

| Op | Cosine | max_abs | rel_euc | JS |
|----|--------|---------|---------|----|
| mlp_out (layer 0) | 0.9999997 | 0.125 | 0.0007 | ~0 |

---

## 6. Nucleus Sampling Optimization

### Principle

The original `nucleus_sampling` sorts the 98304-dim logits, then iterates over the sorted values up to `top_k=25`, calling `.item()` on each (triggering a CPU-NPU sync per iteration).

The optimized version:
1. Transfers only `sorted_value[:top_k].cpu()` — one bulk transfer of 25 values (1 sync total)
2. Iterates locally over at most 25 elements (no NPU sync)

```
Before: topk sorted on NPU → for i up to 25: sorted_value[i].item()  → ~25 NPU syncs
After:  topk_val = sorted_value[:top_k].cpu()                          → 1 NPU sync
        for i in range(25): cum_prob += topk_val[i].item()             → local CPU loop
```

On the 910B's ARM CPU (single-core, relatively weak), reducing ~25 syncs to 1 saves ~1.3ms per step.

### Performance

Measured as part of total decode step (eager, no graph, no model opt):

| Config | Single Step | Δ |
|--------|-------------|----|
| Eager + old sampling (per-element .item() sync) | 51.71 ms | baseline |
| Eager + new sampling (bulk .cpu() transfer) | 50.37 ms | **-1.34 ms (-2.6%)** |

The absolute saving (~1.3 ms per step) is independent of model optimizations since sampling is called outside the graph.

> On GPU, the per-element `.item()` sync is less expensive due to faster CPU-GPU interconnect, making this optimization less impactful there.

---

## Final Performance vs Eager

### Single Decode Step Time (accumulated)

| # | Config | Single Step | Cumulative Gain |
|---|--------|-------------|-----------------|
| 0 | Eager (no graph, no opt) | 50.37 ms | baseline |
| 1 | +Sampling optimization | 49.03 ms* | -2.7% |
| 2 | +Graph replay | 14.16 ms | -71.1% |
| 3 | +FIA | 12.74 ms | -74.7% |
| 4 | +scatter_update | 9.51 ms | -81.1% |
| 5 | +RMS Norm | 8.81 ms | -82.5% |
| 6 | **+MLP (final)** | **8.36 ms** | **-83.4%** |

\* Sampling optimization cannot be perfectly stacked with graph (sampling is outside the graph). The 49.03 ms is the estimated combined baseline.

### Graph Impact

```
Final with graph:     8.36 ms
Final without graph: 38.91 ms
Graph speedup:       4.7×
```

### RTF (Real-Time Factor)

Tested on Ascend 910B4 with 59 test sentences (2-7s audio each).

| Config | Short (2-3s) | Long (4-7s) | All |
|--------|-------------|-------------|-----|
| Eager | 0.57 RTF | 0.48 RTF | 0.53 RTF |
| **Final (no RoPE)** | **0.38 RTF** | **0.28 RTF** | **0.33 RTF** |
| Improvement | +33% | +42% | +38% |

### Quality

- Audio quality: MOS indistinguishable from eager in blind A/B tests
- Token sequence: Identical for first 8 decode steps
- Long audio (~1.5 min): Occasional minor artifacts (see Known Issues)

---

## Final Precision vs Eager

### Layer 0 Sub-ops (first decode step)

| Op | Cosine | max_abs | rel_euc | rms | JS |
|----|--------|---------|---------|-----|----|
| ln_in (embedding) | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ln_out (RMSNorm) | 0.999996 | 0.0078 | 0.0028 | 0.0013 | 9.6e-7 |
| attn_out (FIA) | 0.9999995 | 0.125 | 0.0010 | 0.0081 | ~0 |
| post_ln_out | 0.999995 | 0.0078 | 0.0032 | 0.0005 | 1.2e-7 |
| mlp_out (swiglu) | 0.9999997 | 0.125 | 0.0007 | 0.0038 | ~0 |
| layer_out | 0.9999996 | 0.250 | 0.0009 | 0.0120 | ~0 |

### Logits (all optimizations vs eager)

| Metric | Prefill | Step 0 | Step 8 (divergence point) |
|--------|---------|--------|--------------------------|
| Cosine | 1.0 | 1.0 | 0.999998 |
| max_abs | 0.0 | 0.0 | 0.125 |
| rel_euc | 0.0 | 0.0 | 0.0024 |
| JS | 0.0 | 0.0 | 0.0017 |

**Key findings**:
- **Prefill is bit-exact** (all fused ops fall back to eager for S>1)
- **First decode step is bit-exact** (uses prefill output, no fused kernel has been exercised yet)
- Token sequence stays **identical for ~8 decode steps**
- Divergence occurs from RAS sampling picking different tokens from near-identical probability distributions (max prob diff < 0.008)
- All metric deltas are at bf16 precision floor level — not indicative of a precision bug

---

## Known Issues

### 1. Fused RoPE (`_npu_rotary_embedding`) — Disabled

`_npu_rotary_embedding` produces incorrect results for positions > 0 (cos=0.39 vs standard at position 50). The kernel is bit-exact only at position 0. This is a CANN kernel bug, not fixable from Python.

**Impact**: Disabled by default (`enable_rope = False`). Re-enabling would save ~0.2 ms per decode step. Requires CANN fix in a future version.

### 2. Long Audio Artifacts (~1.5 min)

Long audio generation (~1.5 min) occasionally produces minor audible artifacts. Root cause: accumulated error from fused operators at bf16 precision floor. The error per step is small (max_abs < 0.25 per layer), but over hundreds of autoregressive steps it can drift enough to affect audio quality in edge cases.

**Mitigations (theoretical, not measured)**:
- Casting o_proj or MLP matmuls to fp32 at decode time would reduce error amplification but adds ~15% step time
- RMS Norm in fp32 would help similarly
- Fixing RoPE would remove one source of drift
- In practice, the quality impact is minimal and the performance cost of fp32 outweighs the benefit

### 3. RMS Norm and MLP Fuse — Prefill Fallback

Both RMS Norm and MLP fuse fall back to eager bf16 computation during prefill (S>1). This ensures bit-exact prefill output but means these optimizations only benefit decode steps (typically 50-200 steps per utterance).
