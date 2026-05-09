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
| `GLMTTS_DISABLE_RMS_NORM=1` | Fused RMS Norm | `npu_rms_norm` replacing pow+mul (decode only, S=1) |
| `GLMTTS_DISABLE_ROPE=1` | Fused RoPE | `_npu_rotary_embedding` with pre-computed cos/sin cache (see precision note below) |
| `GLMTTS_DISABLE_SCATTER_UPDATE=1` | KV Cache Write | `scatter_update_` replacing `index_copy_` for decode S=1 |
| `GLMTTS_DISABLE_MLP_FUSE=1` | Fused MLP | gate_up + `npu_swiglu` (single linear + fused activation, decode only) |
| `GLMTTS_DISABLE_DST_SAMPLING=1` | Vectorized Sampling | `dst_sampling` replacing `nucleus_sampling` Python for-loop |

## Optimization Details

### 1. Fused Inference Attention (FIA)

Replaces eager attention with `npu_fused_infer_attention_score` for decode step (batch=1, seq=1). Prefill still uses eager attention.

- Uses BNSD layout, sparse_mode=0 with bool attention mask
- Pre-allocates output/workspace buffers for NPUGraph compatibility
- **Precision**: maxdiff ~0.0625 per token (bf16 ULP level), no error accumulation across decode steps

### 2. Fused RMS Normalization (decode only)

Replaces `torch.pow(hidden_states, 2).mean(-1, keepdim=True) + eps` → `torch.rsqrt` → `hidden_states * weight` with a single `npu_rms_norm` kernel call.

- **Decode only (S=1)**: prefill falls back to eager float32 computation for numerical accuracy
- **Precision**: maxdiff ~0.0078 per layer (bf16 ULP level)

### 3. Fused Rotary Position Embedding ⚠️

Replaces the standard `apply_rotary_pos_emb` (cos/sin multiply + rotate_half) with `_npu_rotary_embedding`.

- **Disabled by default** due to bf16 precision divergence: causes token sampling divergence within 1-2 decode steps
- ~40% kernel scheduling overhead for negligible compute gain
- To enable: set `GLMTTS_NPU_OPT=1` (default) and do NOT set `GLMTTS_DISABLE_ROPE` — but see precision caveat
- TODO: investigate CANN kernel improvements in future versions

### 4. scatter_update_ KV Cache Write

Replaces `index_copy_` in `StaticLayer.update` with `torch_npu.scatter_update_` for decode (S=1).

- ~2.7x faster for single-token KV cache update
- Prefill (S>1) unaffected — falls back to original `index_copy_`
- **Precision**: zero error (bit-exact)

### 5. Fused MLP (gate_up + npu_swiglu, decode only)

Concatenates `gate_proj` and `up_proj` weights into a single `gate_up_weight` parameter, then:

1. Single `F.linear` for both gate and up projections
2. `npu_swiglu` fuses SiLU activation + element-wise multiply

- **Decode only (S=1)**: prefill falls back to eager `silu(gate) * up` for numerical accuracy
- **Precision**: maxdiff ~0.125 per layer (bf16 ULP level), comparable to bf16 computation variance

### 6. Vectorized Nucleus Sampling (dst_sampling)

Replaces the Python for-loop in `nucleus_sampling` with vectorized PyTorch ops:

- Sort + cumsum + mask → top-k/top-p filtering in one pass
- **bf16 cumsum fix**: casts to float32 before cumsum (CANN bf16 cumsum is ~70x slower)
- 3.3x faster than original Python loop
- Sampling distribution verified: 200/200 exact token match vs nucleus_sampling

## Precision Verification

Detailed layer-by-layer comparison between optimized (no RoPE) and eager (no optimizations):

### First decode step (layer 0 sub-ops)

| Op | Cosine | max_abs | rel_euc | JS |
|---|---|---|---|---|
| ln_in (embedding) | 1.0 | 0.0 | 0.0 | 0.0 |
| ln_out (RMSNorm) | 0.999996 | 0.0078 | 0.0028 | 9.6e-7 |
| attn_out (FIA) | 0.9999995 | 0.125 | 0.0010 | ~0 |
| mlp_out (swiglu) | 0.9999997 | 0.125 | 0.0007 | ~0 |
| layer_out | 0.9999996 | 0.25 | 0.0009 | ~0 |

### Logits comparison (step 8, where tokens diverge)

| Metric | Value |
|---|---|
| Cosine | 0.999996 |
| max_abs | 0.125 (1 bf16 ULP) |
| rel_euc | 0.0042 |
| JS | 0.00005 |

**Key finding**: All differences are at bf16 precision floor level. The token sequence stays identical for the first ~8 decode steps, after which RAS sampling may pick different tokens from near-identical probability distributions (max prob diff < 0.008). This is inherent to bf16 computation path differences in fused kernels and does not indicate a precision bug.

## Performance Reference

Tested on Ascend 910B4 with 60 test sentences (2-7s audio each). Final config: NPUGraph + FIA + scatter_update + RMS_norm(decode) + MLP_fuse(decode). RoPE disabled (see precision note).

| Category | Items | Avg RTF | Avg inference |
|---|---|---|---|
| Short (2-3s audio) | 29 | 0.38 | 0.84s |
| Long (4-7s audio) | 30 | 0.28 | 1.60s |
| **All** | **59** | **0.33** | **1.22s** |

Long text (97s audio, single chunk): **RTF=0.28**.

### Performance breakdown by optimization

| Config | RTF (short) | RTF (long) | vs eager |
|---|---|---|---|
| Eager (no graph, no fused) | 0.57 | 0.48 | baseline |
| Graph + all fused incl. RoPE | 0.40 | 0.32 | +29% |
| **Graph + all fused, no RoPE** | **0.38** | **0.28** | **+40%** |

## Known Issues

- **RoPE**: `_npu_rotary_embedding` causes token divergence within 1-2 decode steps due to bf16 precision differences with the eager rotation implementation. Disabled by default. The ~40% kernel scheduling overhead is not worth the precision cost.
- **RMS Norm** and **MLP fuse**: Both fall back to eager float32 computation during prefill (S>1) to maintain bit-exact prefill output. Only decode steps use the fused kernels.
