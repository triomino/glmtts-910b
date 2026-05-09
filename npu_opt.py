"""
Ascend NPU optimization patches for GLM-TTS.
============================================

All NPU-specific performance optimizations are consolidated here.
They can be controlled via a single master switch or toggled individually.

Master Switch
-------------
``GLMTTS_NPU_OPT=0|1``  (default: 1 when torch_npu is available)

Set to ``0`` to disable **all** NPU optimizations at once.

Individual Switches
-------------------
Each optimization can be independently disabled by setting the corresponding
``GLMTTS_DISABLE_*`` environment variable to ``1`` or ``true``.

+-------------------------------+----------------------------------------------+--------------------------------------------+
| Env Var                       | Optimization                                 | What it does                               |
+===============================+==============================================+============================================+
| GLMTTS_DISABLE_FIA            | npu_fused_infer_attention_score              | Fused attention kernel for decode (S=1)    |
+-------------------------------+----------------------------------------------+--------------------------------------------+
| GLMTTS_DISABLE_RMS_NORM       | npu_rms_norm                                 | Fused RMS normalization                    |
+-------------------------------+----------------------------------------------+--------------------------------------------+
| GLMTTS_DISABLE_ROPE           | _npu_rotary_embedding                        | Fused Rotary Position Embedding            |
+-------------------------------+----------------------------------------------+--------------------------------------------+
| GLMTTS_DISABLE_SCATTER_UPDATE | scatter_update_ KV cache                     | Faster KV cache write for decode (S=1)     |
+-------------------------------+----------------------------------------------+--------------------------------------------+
| GLMTTS_DISABLE_MLP_FUSE       | gate_up + npu_swiglu                         | Fused MLP gate/up projection + activation  |
+-------------------------------+----------------------------------------------+--------------------------------------------+
| GLMTTS_DISABLE_DST_SAMPLING   | dst_sampling (in cosyvoice/utils/common.py)  | Vectorized nucleus sampling + bf16 cumsum  |
+-------------------------------+----------------------------------------------+--------------------------------------------+

Usage
-----
No explicit initialization is needed — optimizations are applied automatically
when ``load_models()`` is called from ``glmtts_inference.py``.

To disable everything::

    GLMTTS_NPU_OPT=0 python3 glmtts_inference.py ...

To disable a single optimization::

    GLMTTS_DISABLE_ROPE=1 python3 glmtts_inference.py ...
"""

import os
import logging
import torch

try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False


def _is_enabled(flag_name):
    """Check whether a specific NPU optimization is enabled.

    Returns False if:
      - torch_npu is not available
      - GLMTTS_NPU_OPT is set to 0/false (master switch off)
      - GLMTTS_DISABLE_<flag_name> is set to 1/true
    """
    if not NPU_AVAILABLE:
        return False
    master = os.environ.get("GLMTTS_NPU_OPT", "1").strip().lower()
    if master in ("0", "false"):
        return False
    return os.environ.get(f"GLMTTS_DISABLE_{flag_name}", "").strip().lower() not in ("1", "true")


# ---------------------------------------------------------------------------
# Internal state for idempotent patching
# ---------------------------------------------------------------------------
_fia_state = {}
_rope_rms_state = {}
_mlp_fuse_state = {}
_scatter_update_patched = False


# ---------------------------------------------------------------------------
# 1. FIA — npu_fused_infer_attention_score
#    Replaces eager attention for decode step (bs=1, seq=1) with a single
#    fused CANN kernel. Fallback to eager for prefill (seq > 1).
#    Uses BNSD layout, sparse_mode=1 (causal), pre-allocated output/workspace
#    buffers for NPUGraph compatibility.
# ---------------------------------------------------------------------------
def apply_npu_fused_attention(llm):
    if not _is_enabled("FIA"):
        return False
    try:
        import torch_npu
        from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    except ImportError:
        return False

    config = llm.llama.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    scale = head_dim ** -0.5

    ALL_MASK_ATTENTION_FUNCTIONS.register("npu_fused", ALL_MASK_ATTENTION_FUNCTIONS["eager"])

    def fused_fwd(module, q, k, v, mask, scaling, dropout=0.0, **kw):
        # Only use fused kernel for single-token decode; prefill falls back to eager
        bs, nq, ql, hd = q.shape
        if ql != 1 or bs != 1:
            return eager_attention_forward(module, q, k, v, mask, scaling, dropout, **kw)

        layer_idx = module.layer_idx
        max_cl = _fia_state.get("max_cache_len") or k.shape[2]
        ob = _fia_state.get("output_bufs", {})
        lb = _fia_state.get("lse_bufs", {})
        ws = _fia_state.get("workspace")
        attn_mask_buf = _fia_state.get("attn_mask_buf")

        # Lazy-allocate per-layer output buffers on first use
        if layer_idx not in ob:
            dev = q.device
            dt = q.dtype
            ob[layer_idx] = torch.zeros(1, num_heads, 1, head_dim, dtype=dt, device=dev)
            lb[layer_idx] = torch.zeros(1, dtype=torch.float32, device=dev)

        # sparse_mode=0: external bool mask controls which KV positions to attend to.
        # The mask buffer is updated each step by _prepare_step_buffers.
        if ws is None:
            _mask_sample = torch.ones(1, 1, 1, max_cl, dtype=torch.bool, device=q.device)
            ws = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=torch.randn(1, num_heads, 1, head_dim, dtype=q.dtype, device=q.device),
                key=torch.randn(1, num_kv_heads, max_cl, head_dim, dtype=q.dtype, device=q.device),
                value=torch.randn(1, num_kv_heads, max_cl, head_dim, dtype=q.dtype, device=q.device),
                input_layout="BNSD",
                atten_mask=_mask_sample,
                actual_seq_lengths=[1],
                actual_seq_lengths_kv=[max_cl],
                num_key_value_heads=num_kv_heads, num_heads=num_heads,
                scale=scale, sparse_mode=0,
            )
            _fia_state["workspace"] = ws

        # NPUGraph task grouping for correct stream synchronization
        stream = torch_npu.npu.current_stream()
        in_capture = _fia_state.get("in_graph_capture", False)
        if in_capture:
            torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=q, key=k, value=v,
            input_layout="BNSD",
            atten_mask=attn_mask_buf if attn_mask_buf is not None else None,
            actual_seq_lengths=[1],
            actual_seq_lengths_kv=[max_cl],
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
            workspace=ws,
            out=[ob[layer_idx], lb[layer_idx]],
        )
        if in_capture:
            torch.npu.graph_task_group_end(stream)

        return ob[layer_idx].transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["npu_fused"] = fused_fwd
    config._attn_implementation = "npu_fused"
    for layer in llm.llama.model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "npu_fused"

    logging.info("[npu_opt] FIA enabled (npu_fused_infer_attention_score, BNSD, sparse_mode=0, bool attn_mask)")
    return True


def set_fia_max_cache_len(max_cache_len):
    _fia_state["max_cache_len"] = max_cache_len


def set_fia_graph_capture(in_capture: bool):
    _fia_state["in_graph_capture"] = in_capture


def update_fia_state(**kwargs):
    _fia_state.update(kwargs)


def get_fia_state():
    return _fia_state


# ---------------------------------------------------------------------------
# 2+3. RMS Norm + RoPE
#     npu_rms_norm: single fused kernel replacing torch.pow + mul.
#     _npu_rotary_embedding: fused RoPE with pre-computed cos/sin cache.
#     cos_sin_cache is sized to max(bucket_sizes) to avoid GatherV3 OOB
#     when using NPUGraph with multiple bucket runners.
# ---------------------------------------------------------------------------
def apply_npu_rope_rms(llm, max_cache_len, position_ids_buf):
    enable_rope = False  # TODO: _npu_rotary_embedding causes token divergence within 1-2 decode steps due to bf16 precision differences; ~40% kernel scheduling overhead but negligible compute gain
    enable_rms = _is_enabled("RMS_NORM")
    if not enable_rms and not enable_rope:
        return False
    try:
        import torch_npu
        from transformers.models.llama import modeling_llama as llama_mod
    except ImportError:
        return False

    # On subsequent bucket runners, just update the position buffer
    if _rope_rms_state.get("patched"):
        _rope_rms_state["positions_buf"] = position_ids_buf.view(-1)
        return True

    # Size cos_sin_cache to the largest bucket to prevent OOB in NPUGraph
    bucket_sizes = getattr(llm, "hf_graph_buckets", None) or [max_cache_len]
    actual_max = max(max(bucket_sizes), max_cache_len)

    config = llm.llama.config
    head_dim = config.head_dim
    dev = position_ids_buf.device
    dtype = llm.llama.model.embed_tokens.weight.dtype
    inv_freq = llm.llama.model.rotary_emb.inv_freq.to(dev)
    t = torch.arange(actual_max, dtype=torch.float32, device=dev)
    freqs = torch.outer(t, inv_freq.float())
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)

    _rope_rms_state["cos_sin_cache"] = cos_sin_cache
    _rope_rms_state["positions_buf"] = position_ids_buf.view(-1)
    _rope_rms_state["head_dim"] = head_dim
    _rope_rms_state["patched"] = True

    # Patch LlamaRMSNorm.forward with npu_rms_norm (decode only, S=1)
    if enable_rms:
        _rope_rms_state["_orig_rms_forward"] = llama_mod.LlamaRMSNorm.forward
        def npu_rms_forward(self, hidden_states):
            if hidden_states.shape[1] > 1:  # prefill → eager for precision
                return _rope_rms_state["_orig_rms_forward"](self, hidden_states)
            result, _ = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)
            return result
        llama_mod.LlamaRMSNorm.forward = npu_rms_forward
        logging.info("[npu_opt] npu_rms_norm enabled (decode only, S=1)")

    # Patch apply_rotary_pos_emb with _npu_rotary_embedding for decode (bs=1, seq=1)
    if enable_rope:
        _rope_rms_state["_orig_apply_rotary"] = llama_mod.apply_rotary_pos_emb

        def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            state = _rope_rms_state
            pos_buf = state.get("positions_buf")
            cache = state.get("cos_sin_cache")
            hd = state.get("head_dim")
            if (pos_buf is not None and cache is not None and hd is not None
                    and q.dim() == 4 and q.shape[0] == 1 and q.shape[2] == 1):
                q_flat = q.transpose(1, 2).contiguous().view(1, -1)
                k_flat = k.transpose(1, 2).contiguous().view(1, -1)
                torch_npu._npu_rotary_embedding(pos_buf, q_flat, k_flat, hd, cache, True)
                bs, nq, sl, _ = q.shape
                _, nk, _, _ = k.shape
                return (q_flat.view(bs, sl, nq, hd).transpose(1, 2),
                        k_flat.view(bs, sl, nk, hd).transpose(1, 2))
            return state["_orig_apply_rotary"](q, k, cos, sin, position_ids, unsqueeze_dim)

        llama_mod.apply_rotary_pos_emb = npu_apply_rotary_pos_emb
        logging.info("[npu_opt] _npu_rotary_embedding enabled (cos_sin_cache len=%d)", actual_max)

    return True


# ---------------------------------------------------------------------------
# 4. scatter_update_ KV cache
#    Replaces index_copy_ in StaticLayer.update with torch_npu.scatter_update_
#    for decode step (S=1). Prefill (S>1) falls back to the original path.
# ---------------------------------------------------------------------------
def apply_npu_scatter_update():
    global _scatter_update_patched
    if _scatter_update_patched:
        return
    if not _is_enabled("SCATTER_UPDATE"):
        return
    try:
        import torch_npu
        from transformers.cache_utils import StaticLayer
    except ImportError:
        return

    _orig_layer_update = StaticLayer.update

    def _patched_layer_update(self, key_states, value_states, cache_kwargs=None):
        # Only patch the decode path (batch and seq dim both = 1)
        if key_states.dim() == 4 and key_states.shape[2] == 1:
            cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
            if cache_position is not None:
                if not self.is_initialized:
                    self.lazy_initialization(key_states)
                torch_npu.scatter_update_(self.keys, cache_position, key_states, axis=2)
                torch_npu.scatter_update_(self.values, cache_position, value_states, axis=2)
                return self.keys, self.values
        return _orig_layer_update(self, key_states, value_states, cache_kwargs)

    StaticLayer.update = _patched_layer_update
    _scatter_update_patched = True
    logging.info("[npu_opt] scatter_update_ KV cache patch enabled (decode S=1)")


# ---------------------------------------------------------------------------
# 5. MLP gate_up + npu_swiglu
#    Fuses gate_proj and up_proj into a single linear (gate_up_weight),
#    then applies npu_swiglu which combines silu + element-wise mul in one
#    kernel. Saves one kernel launch for the split and one for mul.
# ---------------------------------------------------------------------------
def apply_npu_mlp_fuse(llm):
    if not _is_enabled("MLP_FUSE"):
        return False
    try:
        import torch_npu
        import torch.nn.functional as F
        from transformers.models.llama import modeling_llama as llama_mod
    except ImportError:
        return False

    if _mlp_fuse_state.get("patched"):
        return True

    config = llm.llama.config
    intermediate_size = config.intermediate_size
    has_bias = getattr(config, "mlp_bias", False)

    # Concatenate gate_proj and up_proj weights into a single parameter
    for layer in llm.llama.model.layers:
        mlp = layer.mlp
        gate_w = mlp.gate_proj.weight.data
        up_w = mlp.up_proj.weight.data
        gate_up_w = torch.cat([gate_w, up_w], dim=0)
        mlp._gate_up_weight = torch.nn.Parameter(gate_up_w)
        if has_bias:
            mlp._gate_bias = mlp.gate_proj.bias.data.clone()
            mlp._up_bias = mlp.up_proj.bias.data.clone()
        else:
            mlp._gate_bias = None
            mlp._up_bias = None

    _orig_forward = llama_mod.LlamaMLP.forward

    def fused_mlp_forward(self, x):
        if x.shape[1] > 1:  # prefill → eager for precision
            return _orig_forward(self, x)
        gu = F.linear(x, self._gate_up_weight)
        if self._gate_bias is not None:
            h = self._gate_up_weight.shape[0] // 2
            g = gu[..., :h] + self._gate_bias
            u = gu[..., h:] + self._up_bias
            gu = torch.cat([g, u], dim=-1)
        return self.down_proj(torch_npu.npu_swiglu(gu))

    llama_mod.LlamaMLP.forward = fused_mlp_forward
    _mlp_fuse_state["patched"] = True
    logging.info("[npu_opt] MLP gate_up + npu_swiglu fuse enabled (%d layers, intermediate=%d)",
                 len(llm.llama.model.layers), intermediate_size)
    return True


# ---------------------------------------------------------------------------
# 6. dst_sampling — vectorized nucleus sampling with bf16 cumsum fix.
#    Implemented in cosyvoice/utils/common.py.
#    ras_sampling() reads GLMTTS_DISABLE_DST_SAMPLING at call time and
#    dispatches to either dst_sampling (vectorized) or nucleus_sampling
#    (original Python for-loop).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------
def apply_all_npu_optimizations(llm):
    """Apply model-level NPU optimizations (FIA + scatter_update).

    Called from load_models() in glmtts_inference.py.
    RoPE/RMS/MLP optimizations are deferred to graph runner initialization.
    """
    if not NPU_AVAILABLE:
        logging.info("[npu_opt] torch_npu not available, skipping all NPU optimizations")
        return False

    master = os.environ.get("GLMTTS_NPU_OPT", "1").strip().lower()
    if master in ("0", "false"):
        logging.info("[npu_opt] GLMTTS_NPU_OPT=0, all NPU optimizations disabled")
        return False

    applied = 0
    if apply_npu_fused_attention(llm):
        applied += 1
    if apply_npu_scatter_update():
        applied += 1
    logging.info("[npu_opt] %d model-level optimizations applied (rope/rms/mlp deferred to graph runner init)", applied)
    return True


def apply_graph_runner_optimizations(llm, max_cache_len, position_ids_buf):
    """Apply graph-runner-level NPU optimizations (RoPE/RMS + MLP fuse).

    Called from HFNpuGraphDecodeRunner.__init__() for each bucket.
    """
    applied = 0
    if apply_npu_rope_rms(llm, max_cache_len, position_ids_buf):
        applied += 1
    if apply_npu_mlp_fuse(llm):
        applied += 1
    logging.info("[npu_opt] %d graph-runner optimizations applied", applied)
    return applied > 0
