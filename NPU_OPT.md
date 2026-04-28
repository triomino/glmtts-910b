# NPU Performance Optimization Guide

GLM-TTS includes a set of Ascend NPU-specific optimizations for Huawei 910B series chips, consolidated in `npu_opt.py`.

## Quick Start

All optimizations are **enabled by default** when running on Ascend NPU. No configuration needed.

To **disable all** NPU optimizations (e.g. for debugging/baseline comparison):

```bash
GLMTTS_NPU_OPT=0 python3 glmtts_inference.py --use_phoneme --hf_graph_decode
```

## Master Switch

| Env Var | Default | Description |
|---------|---------|-------------|
| `GLMTTS_NPU_OPT` | `1` | Enable (`1`) or disable (`0`) all NPU optimizations |

## Individual Switches

Each optimization can be independently disabled:

| Env Var | Optimization | Description |
|---------|-------------|-------------|
| `GLMTTS_DISABLE_FIA=1` | Fused Attention | `npu_fused_infer_attention_score` for decode S=1 |
| `GLMTTS_DISABLE_RMS_NORM=1` | Fused RMS Norm | `npu_rms_norm` replacing pow+mul |
| `GLMTTS_DISABLE_ROPE=1` | Fused RoPE | `_npu_rotary_embedding` with pre-computed cos/sin cache |
| `GLMTTS_DISABLE_SCATTER_UPDATE=1` | KV Cache Write | `scatter_update_` replacing `index_copy_` for decode S=1 |
| `GLMTTS_DISABLE_MLP_FUSE=1` | Fused MLP | gate_up + `npu_swiglu` (single linear + fused activation) |
| `GLMTTS_DISABLE_DST_SAMPLING=1` | Vectorized Sampling | `dst_sampling` replacing `nucleus_sampling` Python for-loop |

## Optimization Details

### 1. Fused Inference Attention (FIA)

Replaces eager attention with `npu_fused_infer_attention_score` for decode step (batch=1, seq=1). Prefill still uses eager attention.

- Uses BNSD layout, sparse_mode=1 (causal mask)
- Pre-allocates output/workspace buffers for NPUGraph compatibility
- ~17% decode speedup

### 2. Fused RMS Normalization

Replaces `torch.pow(hidden_states, 2).mean(-1, keepdim=True) + eps` → `torch.rsqrt` → `hidden_states * weight` with a single `npu_rms_norm` kernel call.

- Exact numerical match (diff=0) vs PyTorch eager
- ~10-15% reduction in normalization overhead

### 3. Fused Rotary Position Embedding

Replaces the standard `apply_rotary_pos_emb` (cos/sin multiply + rotate_half) with `_npu_rotary_embedding`.

- Pre-computes cos/sin cache sized to `max(bucket_sizes)` to avoid GatherV3 out-of-bounds in NPUGraph
- Only active for decode (bs=1, seq=1); prefill falls back to original

### 4. scatter_update_ KV Cache Write

Replaces `index_copy_` in `StaticLayer.update` with `torch_npu.scatter_update_` for decode (S=1).

- ~2.7x faster for single-token KV cache update
- Prefill (S>1) unaffected — falls back to original `index_copy_`

### 5. Fused MLP (gate_up + npu_swiglu)

Concatenates `gate_proj` and `up_proj` weights into a single `gate_up_weight` parameter, then:

1. Single `F.linear` for both gate and up projections
2. `npu_swiglu` fuses SiLU activation + element-wise multiply

- cos_sim > 0.999999, max_diff ~0.5 (within bf16 precision)
- ~9% MLP speedup

### 6. Vectorized Nucleus Sampling (dst_sampling)

Replaces the Python for-loop in `nucleus_sampling` with vectorized PyTorch ops:

- Sort + cumsum + mask → top-k/top-p filtering in one pass
- **bf16 cumsum fix**: casts to float32 before cumsum (CANN bf16 cumsum is ~70x slower)
- 3.3x faster than original Python loop
- Sampling distribution verified: 200/200 exact token match vs nucleus_sampling

## Performance Reference

Tested on Ascend 910B with `example_zh.jsonl`:

| Category | Sentences | Avg RTF | Min RTF | Max RTF |
|----------|-----------|---------|---------|---------|
| Short (10-15 chars) | 30 | 0.509 | 0.408 | 0.642 |
| Medium (15-20 chars) | 15 | 0.392 | 0.354 | 0.438 |
| Long (20-25 chars) | 15 | 0.442 | 0.326 | 1.088 |
| **All** | **60** | **0.463** | **0.326** | **1.088** |

Long text (85s audio, 11 chunks): RTF=0.466.

## Precision Verification

Greedy decoding with all optimizations ON vs OFF:
- 100/100 token exact match
- Avg cosine similarity: 0.99996
- Max absolute diff: 0.5 (within bf16 range)
- No error accumulation across decode steps
