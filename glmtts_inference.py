# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Authors: Jiayan Cui, Zhihan Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import os
import sys

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_V1"] = "0"
os.environ["VLLM_USE_MODELSCOPE"] = "false"
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

import torch

try:
    torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 0)
    import torch.utils._triton

    if hasattr(torch.utils._triton, "_device_supports_tma"):
        torch.utils._triton._device_supports_tma = (
            lambda *args, **kwargs: False
        )
    if hasattr(torch.utils._triton, "has_triton_experimental_host_tma"):
        torch.utils._triton.has_triton_experimental_host_tma = (
            lambda *args, **kwargs: False
        )
except Exception as e:
    print(f"import failed: {e}")

import torchaudio
import tqdm
import logging
import time
import argparse
import json
import math
from pathlib import Path
from functools import partial

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import file_utils, seed_util
from utils import tts_model_util, yaml_util
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import StaticCache
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from vllm import LLM, SamplingParams
from vllm.sampling_params import SamplingParams as VllmSamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

from npu_opt import (
    NPU_AVAILABLE as _NPU_OPT_AVAILABLE,
    apply_all_npu_optimizations,
    apply_graph_runner_optimizations,
    set_fia_max_cache_len,
    set_fia_graph_capture,
    update_fia_state,
    get_fia_state,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LLM_SEQ_INP_LEN = 750
TOKEN_RATE = 25
EOS_TOKEN_ID_AFTER_MINUS_BOS = None
PROMPT_CACHE = {}

# Configure logging
# Conservative RAS tuning for the restored old vLLM path.
# Goal: reduce breath/noise artifacts without touching chunk/cache behavior.
RAS_TOP_P = 0.8
RAS_TOP_K = 25
RAS_WIN_SIZE = 10
RAS_TAU_R = 0.20
RAS_TEMPERATURE = 0.95

# HF stepwise experiment params to compare against the vLLM path.
HF_SAMPLE_TOP_P = 0.8
HF_SAMPLE_TOP_K = 25
HF_SAMPLE_TEMPERATURE = 0.7
DEFAULT_HF_GRAPH_BUCKETS = "512,1024,1536"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


def _elapsed_seconds(start_time):
    return time.perf_counter() - start_time


def _write_token_dump(
    dump_dir,
    uttid,
    backend,
    sample_method,
    chunk_records,
):
    if not dump_dir:
        return

    safe_uttid = uttid or "interactive"
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    file_path = dump_path / f"{safe_uttid}.{backend}.{sample_method}.tokens.json"
    payload = {
        "uttid": safe_uttid,
        "backend": backend,
        "sample_method": sample_method,
        "num_chunks": len(chunk_records),
        "chunks": chunk_records,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("[token_dump] wrote %s", file_path)


def set_npu_jit(enabled: bool):
    if NPU_AVAILABLE and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_compile_mode(jit_compile=enabled)


if NPU_AVAILABLE:
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    # 备份原函数
    _orig_torch_fbank = WhisperFeatureExtractor._torch_extract_fbank_features
    _orig_hann_window = torch.hann_window

    def _torch_fbank_cpu_only(self, waveform, device="cpu"):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().to("cpu", dtype=torch.float32)
        return _orig_torch_fbank(self, waveform, device="cpu")

    def _hann_window_force_fp32(window_length, *args, **kwargs):
        kwargs["dtype"] = torch.float32
        return _orig_hann_window(window_length, *args, **kwargs)

    torch.hann_window = _hann_window_force_fp32
    WhisperFeatureExtractor._torch_extract_fbank_features = _torch_fbank_cpu_only


def get_special_token_ids(tokenize_fn):
    """
    Get special token IDs based on the tokenizer name.
    """
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    special_token_ids = {}

    # Validation
    endoftext_id = tokenize_fn("<|endoftext|>")[0]
    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        # Check 1: Special token length must be 1
        if len(__ids) != 1:
            raise AssertionError(
                f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}"
            )
        # Check 2: Special token ID must be >= endoftext_id
        if __ids[0] < endoftext_id:
            raise AssertionError(
                f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}"
            )

        special_token_ids[k] = __ids[0]

    return special_token_ids

def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len

def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend"):
    if sample_rate == 32000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=640, n_fft=2560, num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False)
        logging.info("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)
        logging.info("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join('ckpt', 'vq32k-phoneme-tokenizer'), trust_remote_code=True
    )

    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend

class VllmRasLogitsProcessor(AdapterLogitsProcessor):
    """Approximate the original GLMTTS RAS behavior inside vLLM."""

    @classmethod
    def validate_params(cls, params: VllmSamplingParams):
        extra_args = params.extra_args or {}
        if not extra_args.get("ras_enabled", False):
            return None

        top_p = extra_args.get("ras_top_p", RAS_TOP_P)
        top_k = extra_args.get("ras_top_k", RAS_TOP_K)
        win_size = extra_args.get("ras_win_size", RAS_WIN_SIZE)
        tau_r = extra_args.get("ras_tau_r", RAS_TAU_R)
        temperature = extra_args.get("ras_temperature", RAS_TEMPERATURE)

        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"ras_top_p must be in (0, 1], got {top_p}")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"ras_top_k must be a positive integer, got {top_k}")
        if not isinstance(win_size, int) or win_size <= 0:
            raise ValueError(
                f"ras_win_size must be a positive integer, got {win_size}"
            )
        if tau_r < 0:
            raise ValueError(f"ras_tau_r must be >= 0, got {tau_r}")
        if temperature <= 0:
            raise ValueError(f"ras_temperature must be > 0, got {temperature}")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: VllmSamplingParams,
    ) -> RequestLogitsProcessor | None:
        extra_args = params.extra_args or {}
        if not extra_args.get("ras_enabled", False):
            return None

        top_p = float(extra_args.get("ras_top_p", RAS_TOP_P))
        top_k = int(extra_args.get("ras_top_k", RAS_TOP_K))
        win_size = int(extra_args.get("ras_win_size", RAS_WIN_SIZE))
        tau_r = float(extra_args.get("ras_tau_r", RAS_TAU_R))
        temperature = float(extra_args.get("ras_temperature", RAS_TEMPERATURE))
        repeat_threshold = max(1, math.ceil(win_size * tau_r))

        def ras_logits_processor(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            logits_fp32 = logits.float()
            base_probs = torch.softmax(logits_fp32, dim=-1)

            scaled_logits = logits_fp32 / temperature
            sorted_probs, sorted_idx = torch.softmax(scaled_logits, dim=-1).sort(
                descending=True, stable=True
            )

            selected_probs = []
            selected_indices = []
            cum_prob = 0.0
            for i in range(len(sorted_idx)):
                if cum_prob < top_p and len(selected_probs) < top_k:
                    prob_i = sorted_probs[i]
                    cum_prob += float(prob_i.item())
                    selected_probs.append(prob_i)
                    selected_indices.append(int(sorted_idx[i].item()))
                else:
                    break

            if not selected_probs:
                return logits

            selected_probs_t = torch.stack(selected_probs)
            selected_idx_t = torch.tensor(
                selected_indices, dtype=torch.long, device=logits.device
            )
            selected_probs_t = selected_probs_t / selected_probs_t.sum()

            repeated_mask = torch.zeros_like(selected_probs_t, dtype=torch.bool)
            if output_ids:
                recent_ids = output_ids[-win_size:]
                for idx, token_id in enumerate(selected_indices):
                    if recent_ids.count(token_id) >= repeat_threshold:
                        repeated_mask[idx] = True

            if repeated_mask.any():
                fallback_mass = selected_probs_t[repeated_mask].sum()
                final_probs = fallback_mass * base_probs
                final_probs[selected_idx_t[~repeated_mask]] += selected_probs_t[
                    ~repeated_mask
                ]
            else:
                final_probs = torch.zeros_like(base_probs)
                final_probs[selected_idx_t] = selected_probs_t

            final_probs = final_probs.clamp_min(torch.finfo(final_probs.dtype).tiny)
            return torch.log(final_probs).to(logits.dtype)

        return ras_logits_processor


def _hf_nucleus_sample(
    weighted_scores: torch.Tensor,
    top_p: float = HF_SAMPLE_TOP_P,
    top_k: int = HF_SAMPLE_TOP_K,
    temperature: float = HF_SAMPLE_TEMPERATURE,
) -> torch.Tensor:
    scaled_scores = weighted_scores.float() / temperature
    sorted_value, sorted_idx = scaled_scores.softmax(dim=0).sort(
        descending=True, stable=True
    )
    cum_probs = torch.cumsum(sorted_value, dim=0)
    mask = (cum_probs - sorted_value) < top_p
    if top_k < mask.shape[0]:
        mask[top_k:] = False
    if not mask.any():
        mask[0] = True
    prob = sorted_value[mask]
    indices = sorted_idx[mask]
    return indices[prob.multinomial(1, replacement=True)]


def _build_full_input_ids(
    llm, prompt_text_token, tts_text_token, prompt_speech_token, teacher_tokens
):
    prompt_speech_token_offset = prompt_speech_token + llm.ats
    teacher_abs = [token + llm.ats for token in teacher_tokens]
    return (
        prompt_text_token.squeeze().tolist()
        + tts_text_token.squeeze().tolist()
        + [llm.boa]
        + prompt_speech_token_offset.squeeze().tolist()
        + teacher_abs
    )


class HFNpuGraphDecodeRunner:
    def __init__(self, llm, max_cache_len=1536, warmup_iters=2):
        self.llm = llm
        self.max_cache_len = int(max_cache_len)
        self.warmup_iters = max(1, int(warmup_iters))
        self.device = llm.llama.model.embed_tokens.weight.device
        self.embed_dtype = llm.llama.model.embed_tokens.weight.dtype
        self.hidden_size = llm.llama.model.embed_tokens.weight.shape[1]
        self.mask_fill_value = torch.finfo(self.embed_dtype).min
        self.static_cache = StaticCache(
            config=llm.llama.config,
            max_cache_len=self.max_cache_len,
        )
        self.input_embeds_buf = torch.zeros(
            (1, 1, self.hidden_size),
            dtype=self.embed_dtype,
            device=self.device,
        )
        self.attention_mask_buf = torch.full(
            (1, 1, 1, self.max_cache_len),
            self.mask_fill_value,
            dtype=self.embed_dtype,
            device=self.device,
        )
        self.cache_position_buf = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.position_ids_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.token_id_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.graphed_step = None
        self.current_pos = 0
        self._cache_initialized = False
        self._graph_built = False
        self._use_fia = (
            NPU_AVAILABLE
            and getattr(llm.llama.config, "_attn_implementation", None) == "npu_fused"
        )
        self._npu_graph = None
        self._logits_buf = None
        self.fia_workspace = None
        self.fia_output_bufs = {}
        self.fia_lse_bufs = {}
        if self._use_fia:
            self._init_fia_buffers()
            set_fia_max_cache_len(self.max_cache_len)
        self._use_rope_rms = False
        self._use_mlp_fuse = False
        if apply_graph_runner_optimizations(llm, self.max_cache_len, self.position_ids_buf):
            self._use_rope_rms = True
            self._use_mlp_fuse = True

    def can_handle(self, prefill_len: int, max_new_tokens: int) -> bool:
        return (int(prefill_len) + int(max_new_tokens)) <= self.max_cache_len

    def _init_fia_buffers(self):
        import torch_npu

        config = self.llm.llama.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        n_layers = config.num_hidden_layers
        for i in range(n_layers):
            self.fia_output_bufs[i] = torch.zeros(
                1, num_heads, 1, head_dim, dtype=self.embed_dtype, device=self.device
            )
            self.fia_lse_bufs[i] = torch.zeros(1, dtype=torch.float32, device=self.device)

        q_sample = torch.randn(
            1, num_heads, 1, head_dim, dtype=self.embed_dtype, device=self.device
        )
        k_sample = torch.randn(
            1, num_kv_heads, self.max_cache_len, head_dim, dtype=self.embed_dtype, device=self.device
        )
        v_sample = torch.randn(
            1, num_kv_heads, self.max_cache_len, head_dim, dtype=self.embed_dtype, device=self.device
        )
        self.fia_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query=q_sample,
            key=k_sample,
            value=v_sample,
            input_layout="BNSD",
            actual_seq_lengths_kv=[self.max_cache_len],
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=head_dim ** -0.5,
            sparse_mode=3,
        )
        ws_mb = self.fia_workspace.numel() * self.fia_workspace.element_size() / 1024 / 1024
        logging.info(
            "[fia] buffers allocated: %d layers, workspace=%.0f MB, max_cache_len=%d",
            n_layers, ws_mb, self.max_cache_len,
        )

    def reset(self):
        if self._cache_initialized:
            self.static_cache.reset()
        self.current_pos = 0

    def _step_logits_impl(self, input_embeds):
        position_embeddings = self.llm.llama.model.rotary_emb(
            input_embeds, self.position_ids_buf
        )
        hidden_states = input_embeds
        for decoder_layer in self.llm.llama.model.layers[
            : self.llm.llama.model.config.num_hidden_layers
        ]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=self.attention_mask_buf,
                position_ids=self.position_ids_buf,
                past_key_values=self.static_cache,
                cache_position=self.cache_position_buf,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.llm.llama.model.norm(hidden_states)
        logits = self.llm.llama.lm_head(hidden_states)
        return logits[:, -1, :]

    @torch.inference_mode()
    def _build_graph(self):
        if self._graph_built:
            return
        logging.info(
            "[hf_graph] building static single-step graph max_cache_len=%s warmup_iters=%s fia=%s mlp_fuse=%s",
            self.max_cache_len,
            self.warmup_iters,
            self._use_fia,
            self._use_mlp_fuse,
        )
        if self._use_fia:
            update_fia_state(
                workspace=self.fia_workspace,
                output_bufs=self.fia_output_bufs,
                lse_bufs=self.fia_lse_bufs,
                max_cache_len=self.max_cache_len,
            )
        for _ in range(self.warmup_iters):
            _ = self._step_logits_impl(self.input_embeds_buf)
        torch.npu.synchronize()
        if self._use_fia:
            self._npu_graph = torch.npu.NPUGraph()
            set_fia_graph_capture(True)
            with torch.npu.graph(self._npu_graph):
                self._logits_buf = self._step_logits_impl(self.input_embeds_buf)
            set_fia_graph_capture(False)
        else:
            self.graphed_step = torch.npu.make_graphed_callables(
                self._step_logits_impl,
                (self.input_embeds_buf,),
                num_warmup_iters=self.warmup_iters,
            )
        torch.npu.synchronize()
        self._graph_built = True

    def prefill(self, full_input_ids):
        self.reset()
        self.attention_mask_buf.fill_(self.mask_fill_value)
        prefill_len = len(full_input_ids)
        self.attention_mask_buf[..., :prefill_len] = 0
        input_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=self.device)
        inputs_embeds = self.llm.llama_embedding(input_tensor)
        prefill_last_idx = torch.tensor([prefill_len - 1], dtype=torch.long, device=self.device)
        outputs = self.llm.llama(
            inputs_embeds=inputs_embeds,
            past_key_values=self.static_cache,
            use_cache=True,
            logits_to_keep=prefill_last_idx,
            return_dict=True,
        )
        self._cache_initialized = True
        self.current_pos = int(input_tensor.shape[1])
        return outputs.logits[0, -1].log_softmax(dim=-1)

    def _prepare_step_buffers(self, sampled_abs_token: int):
        self.token_id_buf[0, 0] = sampled_abs_token
        self.input_embeds_buf.copy_(self.llm.llama_embedding(self.token_id_buf))
        self.cache_position_buf[0] = self.current_pos
        self.position_ids_buf[0, 0] = self.current_pos
        self.attention_mask_buf[..., self.current_pos] = 0

    def step(self, sampled_abs_token: int):
        self._prepare_step_buffers(sampled_abs_token)
        if os.environ.get("GLMTTS_NO_GRAPH_REPLAY", "").strip().lower() in ("1", "true"):
            if self._use_fia and not get_fia_state().get("max_cache_len") == self.max_cache_len:
                update_fia_state(
                    workspace=self.fia_workspace,
                    output_bufs=self.fia_output_bufs,
                    lse_bufs=self.fia_lse_bufs,
                    max_cache_len=self.max_cache_len,
                )
            self._logits_buf = self._step_logits_impl(self.input_embeds_buf)
            self.current_pos += 1
            return self._logits_buf[0].log_softmax(dim=-1)
        self._build_graph()
        if self._use_fia and self._npu_graph is not None:
            self._npu_graph.replay()
        else:
            self._logits_buf = self.graphed_step(self.input_embeds_buf)
        self.current_pos += 1
        return self._logits_buf[0].log_softmax(dim=-1)

    @torch.inference_mode()
    def prime_graph(self):
        if self._graph_built:
            return
        self.current_pos = 0
        self.static_cache.reset()
        self.attention_mask_buf.fill_(self.mask_fill_value)
        self.cache_position_buf[0] = 0
        self.position_ids_buf[0, 0] = 0
        self.token_id_buf.zero_()
        self.input_embeds_buf.copy_(self.llm.llama_embedding(self.token_id_buf))
        self.attention_mask_buf[..., 0] = 0
        self._build_graph()
        self.static_cache.reset()
        self.attention_mask_buf.fill_(self.mask_fill_value)
        self.current_pos = 0
        self._cache_initialized = False


def _parse_hf_graph_buckets(hf_graph_buckets, fallback_max_cache_len):
    if hf_graph_buckets is None:
        buckets = []
    elif isinstance(hf_graph_buckets, str):
        buckets = [int(item.strip()) for item in hf_graph_buckets.split(",") if item.strip()]
    else:
        buckets = [int(item) for item in hf_graph_buckets]
    buckets = [bucket for bucket in buckets if bucket > 0]
    if fallback_max_cache_len is not None:
        buckets.append(int(fallback_max_cache_len))
    buckets = sorted(set(buckets))
    if not buckets:
        buckets = [1536]
    return buckets


def _select_hf_graph_bucket(llm, required_cache_len: int):
    bucket_sizes = getattr(llm, "hf_graph_buckets", None) or [
        getattr(llm, "hf_graph_max_cache_len", 1536)
    ]
    for bucket_size in bucket_sizes:
        if int(required_cache_len) <= int(bucket_size):
            return int(bucket_size)
    return None


def _maybe_get_hf_graph_runner(llm, required_cache_len: int | None = None):
    if not getattr(llm, "hf_graph_decode", False):
        return None
    if getattr(llm, "inference_backend", None) != "hf":
        return None
    if not NPU_AVAILABLE or not hasattr(torch, "npu"):
        logging.warning("[hf_graph] torch.npu is unavailable, fallback to eager decode")
        llm.hf_graph_decode = False
        return None
    if not hasattr(torch.npu, "make_graphed_callables"):
        logging.warning(
            "[hf_graph] torch.npu.make_graphed_callables is unavailable, fallback to eager decode"
        )
        llm.hf_graph_decode = False
        return None
    if getattr(llm.llama.config, "_attn_implementation", None) not in ("eager", "npu_fused"):
        logging.warning(
            "[hf_graph] attn_implementation=%s is not graph-safe here; require eager or npu_fused, fallback to eager decode",
            getattr(llm.llama.config, "_attn_implementation", None),
        )
        llm.hf_graph_decode = False
        return None

    if required_cache_len is None:
        bucket_size = max(
            getattr(llm, "hf_graph_buckets", None)
            or [getattr(llm, "hf_graph_max_cache_len", 1536)]
        )
    else:
        bucket_size = _select_hf_graph_bucket(llm, required_cache_len)
        if bucket_size is None:
            return None

    runners = getattr(llm, "hf_graph_runners", None)
    if runners is None:
        runners = {}
        llm.hf_graph_runners = runners

    runner = runners.get(bucket_size)
    if runner is None:
        logging.info(
            "[hf_graph] creating bucket runner max_cache_len=%s warmup_iters=%s",
            bucket_size,
            getattr(llm, "hf_graph_warmup_iters", 2),
        )
        runner = HFNpuGraphDecodeRunner(
            llm=llm,
            max_cache_len=bucket_size,
            warmup_iters=getattr(llm, "hf_graph_warmup_iters", 2),
        )
        runners[bucket_size] = runner

    llm.hf_graph_runner = runner
    return runner


def _maybe_prebuild_hf_graph_runners(llm):
    if not getattr(llm, "hf_graph_decode", False):
        return
    if not getattr(llm, "hf_graph_prebuild_buckets", True):
        logging.info("[hf_graph] skip bucket prebuild (disabled)")
        return
    bucket_sizes = getattr(llm, "hf_graph_buckets", None) or [
        getattr(llm, "hf_graph_max_cache_len", 1536)
    ]
    logging.info("[hf_graph] prebuilding bucketed graph runners buckets=%s", bucket_sizes)
    for bucket_size in bucket_sizes:
        runner = _maybe_get_hf_graph_runner(llm, required_cache_len=bucket_size)
        if runner is None:
            return
        with torch.inference_mode():
            runner.prime_graph()


@torch.inference_mode()
def _hf_stepwise_forward_dynamic(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="nucleus",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
):
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    device = tts_text_token.device
    prompt_speech_token = prompt_speech_token + llm.ats
    boa_tensor = torch.tensor([llm.boa], device=device).unsqueeze(0)
    input_ids = torch.cat(
        [prompt_text_token, tts_text_token, boa_tensor, prompt_speech_token], dim=1
    ).to(torch.long)
    inputs_embeds = llm.llama_embedding(input_ids)

    text_len = int(tts_text_token_len.item())
    min_len = int(text_len * 2)
    max_len = int(text_len * 20)

    out_tokens = []
    past_key_values = None

    for i in range(max_len):
        logits_to_keep = (
            torch.tensor([inputs_embeds.shape[1] - 1], dtype=torch.long, device=device)
            if inputs_embeds.shape[1] > 1
            else 1
        )
        outputs = llm.llama(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=logits_to_keep,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logp = outputs.logits[0, -1].log_softmax(dim=-1)

        if sample_method == "ras":
            if i < min_len:
                logp[llm.eoa] = -float("inf")
            top_ids = llm.sampling_ids_ras(logp, out_tokens, sampling).item()
        elif sample_method == "topk":
            top_ids = llm.sampling_ids(
                logp, sampling, beam_size, ignore_eos=(i < min_len)
            ).item()
        elif sample_method == "nucleus":
            if i < min_len:
                logp[llm.eoa] = -float("inf")
            top_ids = _hf_nucleus_sample(
                logp,
                top_p=sampling_top_p,
                top_k=sampling_top_k,
                temperature=sampling_temperature,
            ).item()
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        if top_ids == llm.eoa:
            break

        out_tokens.append(top_ids)
        inputs_embeds = llm.llama_embedding(
            torch.tensor([[top_ids]], dtype=torch.long, device=device)
        )

    return [token_id - llm.ats for token_id in out_tokens]


@torch.inference_mode()
def _hf_stepwise_forward_static_graph(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="nucleus",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
):
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    text_len = int(tts_text_token_len.item())
    min_len = int(text_len * 2)
    max_len = int(text_len * 20)
    full_input_ids = _build_full_input_ids(
        llm, prompt_text_token, tts_text_token, prompt_speech_token, []
    )
    required_cache_len = len(full_input_ids) + max_len
    runner = _maybe_get_hf_graph_runner(llm, required_cache_len=required_cache_len)
    if runner is None:
        logging.info(
            "[hf_graph] fallback to eager: required_cache_len=%s has no matching bucket in buckets=%s",
            required_cache_len,
            getattr(llm, "hf_graph_buckets", None),
        )
        return None
    if not runner.can_handle(len(full_input_ids), max_len):
        logging.info(
            "[hf_graph] fallback to eager: required_cache_len=%s > max_cache_len=%s. Chunking shorter text can reduce this pressure.",
            required_cache_len,
            runner.max_cache_len,
        )
        return None

    logging.info(
        "[hf_graph] selected bucket max_cache_len=%s required_cache_len=%s prefill_len=%s max_new_tokens=%s",
        runner.max_cache_len,
        required_cache_len,
        len(full_input_ids),
        max_len,
    )

    logp = runner.prefill(full_input_ids)
    out_tokens = []

    for i in range(max_len):
        if sample_method == "ras":
            if i < min_len:
                logp[llm.eoa] = -float("inf")
            top_ids = llm.sampling_ids_ras(logp, out_tokens, sampling).item()
        elif sample_method == "topk":
            top_ids = llm.sampling_ids(
                logp, sampling, beam_size, ignore_eos=(i < min_len)
            ).item()
        elif sample_method == "nucleus":
            if i < min_len:
                logp[llm.eoa] = -float("inf")
            top_ids = _hf_nucleus_sample(
                logp,
                top_p=sampling_top_p,
                top_k=sampling_top_k,
                temperature=sampling_temperature,
            ).item()
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        if top_ids == llm.eoa:
            break

        out_tokens.append(top_ids)
        logp = runner.step(top_ids)

    return [token_id - llm.ats for token_id in out_tokens]


@torch.inference_mode()
def _hf_stepwise_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="nucleus",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
):
    graph_res = _hf_stepwise_forward_static_graph(
        llm=llm,
        prompt_text_token=prompt_text_token,
        tts_text_token=tts_text_token,
        prompt_speech_token=prompt_speech_token,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        sampling_top_p=sampling_top_p,
        sampling_top_k=sampling_top_k,
        sampling_temperature=sampling_temperature,
    )
    if graph_res is not None:
        return graph_res
    return _hf_stepwise_forward_dynamic(
        llm=llm,
        prompt_text_token=prompt_text_token,
        tts_text_token=tts_text_token,
        prompt_speech_token=prompt_speech_token,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        sampling_top_p=sampling_top_p,
        sampling_top_k=sampling_top_k,
        sampling_temperature=sampling_temperature,
    )


def local_llm_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="ras",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
    seed=0,
):
    if getattr(llm, "inference_backend", "hf") == "hf":
        return _hf_stepwise_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            beam_size=beam_size,
            sampling=sampling,
            sample_method=sample_method,
            sampling_top_p=sampling_top_p,
            sampling_top_k=sampling_top_k,
            sampling_temperature=sampling_temperature,
        )

    prompt_speech_token_offset = prompt_speech_token + llm.ats
    input_ids = (
        prompt_text_token.squeeze().tolist()
        + tts_text_token.squeeze().tolist()
        + [llm.boa]
        + prompt_speech_token_offset.squeeze().tolist()
    )

    text_len = tts_text_token.shape[1]
    use_ras = sample_method == "ras"
    allowed_token_ids = [llm.eoa] + list(range(llm.ats, llm.ate + 1))
    sampling_params = SamplingParams(
        temperature=RAS_TEMPERATURE if use_ras else sampling_temperature,
        top_p=1.0 if use_ras else 0.8,
        top_k=0 if use_ras else 25,
        seed=seed,
        max_tokens=int(text_len * 20),
        min_tokens=int(text_len * 2),
        stop_token_ids=[llm.eoa],
        ignore_eos=False,
        allowed_token_ids=allowed_token_ids,
        extra_args={
            "ras_enabled": use_ras,
            "ras_top_p": RAS_TOP_P,
            "ras_top_k": RAS_TOP_K,
            "ras_win_size": RAS_WIN_SIZE,
            "ras_tau_r": RAS_TAU_R,
            "ras_temperature": RAS_TEMPERATURE,
        },
    )

    engine = getattr(llm, "vllm_engine", None)
    if engine is None:
        engine = llm.llm.vllm_engine

    try:
        outputs = engine.generate(
            prompts=[input_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    except TypeError:
        outputs = engine.generate(
            inputs=[{"prompt_token_ids": input_ids}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    generated_ids = outputs[0].outputs[0].token_ids
    while generated_ids and generated_ids[-1] == llm.eoa:
        generated_ids = generated_ids[:-1]
    out_tokens = [t - llm.ats for t in generated_ids]

    return out_tokens


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """
    Single Flow forward pass.
    """
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        n_timesteps=11,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach(), full_mel


def get_prompt_cache_entry(
    frontend, text_frontend, prompt_text_raw, prompt_speech_path, sample_rate
):
    normalized_prompt_text = text_frontend.text_normalize(prompt_text_raw)
    prompt_cache_key = (
        normalized_prompt_text,
        os.path.abspath(prompt_speech_path),
        sample_rate,
    )

    if prompt_cache_key in PROMPT_CACHE:
        return PROMPT_CACHE[prompt_cache_key]

    prompt_text_token = frontend._extract_text_token(normalized_prompt_text + " ")
    prompt_speech_token = frontend._extract_speech_token([prompt_speech_path])
    speech_feat = frontend._extract_speech_feat(
        prompt_speech_path, sample_rate=sample_rate
    )
    embedding = frontend._extract_spk_embedding(prompt_speech_path)
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(DEVICE)

    prompt_entry = {
        "prompt_text": normalized_prompt_text,
        "prompt_text_token": prompt_text_token,
        "prompt_speech_token": prompt_speech_token,
        "speech_feat": speech_feat,
        "embedding": embedding,
        "cache_speech_token": cache_speech_token,
        "flow_prompt_token": flow_prompt_token,
    }
    PROMPT_CACHE[prompt_cache_key] = prompt_entry
    return prompt_entry


# --- Helper Function: Get Prompt from Cache ---
def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    """
    Constructs prompt tokens from the cache.
    Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
    """
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    # Estimate required length ratio
    # Avoid division by zero
    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # Prune cache if too long.
    # Logic: Keep the first item (original prompt), remove from the second item onwards.
    while (
        __len_cache_speech_token() + __len_synth_audi_token_estim
        > MAX_LLM_SEQ_INP_LEN
    ):
        if len(cache_speech_token) <= 1:
            break
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # Construct Text Prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


# --- Main Generation Logic ---
def generate_long(
    frontend: TTSFrontEnd,
    text_frontend: TextFrontEnd,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
    dump_tokens=False,
    token_dump_dir=None,
    flow_prompt_token=None,
    speech_feat=None,
    local_llm_forward=local_llm_forward,
    local_flow_forward=local_flow_forward,
    use_phoneme=False,
):
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    chunk_records = []
    short_text_list = text_frontend.split_by_len(syn_text)
    logging.info(f"[generate_long] uttid={uttid} split into {len(short_text_list)} chunks")

    for chunk_idx, tts_text in enumerate(short_text_list):
        chunk_start = time.perf_counter()
        logging.info(
            f"[generate_long] uttid={uttid} chunk={chunk_idx + 1}/{len(short_text_list)} raw_chars={len(tts_text)}"
        )
        seed_util.set_seed(seed)
        normalize_start = time.perf_counter()
        tts_text_tn = text_frontend.text_normalize(
            tts_text
        )  # Normalize again after splitting
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        normalize_elapsed = _elapsed_seconds(normalize_start)

        text_token_start = time.perf_counter()
        tts_text_token = frontend._extract_text_token(tts_text_tn)
        text_token_elapsed = _elapsed_seconds(text_token_start)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        prompt_prepare_start = time.perf_counter()
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor(
                [cache_speech_token[0]], dtype=torch.int32
            ).to(device)
            logging.debug("[generate_long] Using initial prompt (empty cache history)")
        prompt_prepare_elapsed = _elapsed_seconds(prompt_prepare_start)

        # LLM Inference
        llm_start = time.perf_counter()
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method,
            sampling_top_p=sampling_top_p,
            sampling_top_k=sampling_top_k,
            sampling_temperature=sampling_temperature,
            seed=seed,
        )
        llm_elapsed = _elapsed_seconds(llm_start)

        output_token_list.extend(token_list_res)
        if dump_tokens:
            chunk_records.append(
                {
                    "chunk_idx": chunk_idx,
                    "raw_text": tts_text,
                    "normalized_text": tts_text_tn,
                    "num_tokens": len(token_list_res),
                    "tokens": token_list_res,
                }
            )

        # Flow Inference
        flow_start = time.perf_counter()
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding,
        )
        flow_elapsed = _elapsed_seconds(flow_start)

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

        logging.info(
            f"[generate_long] uttid={uttid} chunk={chunk_idx + 1}/{len(short_text_list)} timings: normalize={normalize_elapsed:.3f}s text_token={text_token_elapsed:.3f}s prompt={prompt_prepare_elapsed:.3f}s llm={llm_elapsed:.3f}s flow={flow_elapsed:.3f}s total={_elapsed_seconds(chunk_start):.3f}s out_tokens={len(token_list_res)} out_samples={output.shape[-1]}"
        )

    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None
    if dump_tokens:
        _write_token_dump(
            dump_dir=token_dump_dir,
            uttid=uttid,
            backend=getattr(llm, "inference_backend", "vllm"),
            sample_method=sample_method,
            chunk_records=chunk_records,
        )

    return tts_speech, tts_mel, output_token_list, text_tn_dict


def jsonl_generate(
    data_name,
    folder_path,
    sample_rate=24000,
    seed=0,
    use_cache=True,
    use_phoneme=False,
    sample_method="ras",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
    dump_tokens=False,
    token_dump_dir=None,
):
    jsonl_path = os.path.join("examples", data_name + ".jsonl")

    # Dataset path resolution
    logging.info(f"Using jsonl: {jsonl_path}")
    item_list = file_utils.get_jsonl(jsonl_path)
    logging.info(f"[jsonl_generate] loaded {len(item_list)} items from {jsonl_path}")

    output_json_path = os.path.join(folder_path, "text_compare.jsonl")

    with open(output_json_path, "w") as f_out:
        for item_idx, item in enumerate(tqdm.tqdm(item_list), start=1):
            try:
                item_start = time.perf_counter()
                uttid = item["uttid"]
                wav_save_path = os.path.join(folder_path, f"{uttid}.wav")

                # Text Normalization
                prompt_prepare_start = time.perf_counter()
                prompt_entry = get_prompt_cache_entry(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    prompt_text_raw=item["prompt_text"],
                    prompt_speech_path=item["prompt_speech"],
                    sample_rate=sample_rate,
                )
                prompt_text = prompt_entry["prompt_text"]
                prompt_text_token = prompt_entry["prompt_text_token"]
                prompt_speech_token = prompt_entry["prompt_speech_token"]
                speech_feat = prompt_entry["speech_feat"]
                embedding = prompt_entry["embedding"]
                cache_speech_token = [
                    tokens.copy() for tokens in prompt_entry["cache_speech_token"]
                ]
                flow_prompt_token = prompt_entry["flow_prompt_token"]
                synth_text = text_frontend.text_normalize(item["syn_text"])
                prompt_prepare_elapsed = _elapsed_seconds(prompt_prepare_start)

                cache = {
                    "cache_text": [prompt_text],
                    "cache_text_token": [prompt_text_token],
                    "cache_speech_token": cache_speech_token,
                    "use_cache": use_cache,
                }
                syn_text = item["syn_text"]
                logging.info(
                    f"[jsonl_generate] item={item_idx}/{len(item_list)} uttid={uttid} prompt_prep={prompt_prepare_elapsed:.3f}s syn_chars={len(syn_text)}"
                )

                # Run Generation
                generate_start = time.perf_counter()
                tts_speech, _, _, text_tn_dict = generate_long(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    llm=llm,
                    flow=flow,
                    text_info=[uttid, synth_text],
                    cache=cache,
                    embedding=embedding,
                    seed=seed,
                    sample_method=sample_method,
                    sampling_top_p=sampling_top_p,
                    sampling_top_k=sampling_top_k,
                    sampling_temperature=sampling_temperature,
                    dump_tokens=dump_tokens,
                    token_dump_dir=token_dump_dir,
                    flow_prompt_token=flow_prompt_token,
                    speech_feat=speech_feat,
                    device=DEVICE,
                    use_phoneme=use_phoneme,
                )
                generate_elapsed = _elapsed_seconds(generate_start)

                f_out.write(
                    json.dumps(text_tn_dict, ensure_ascii=False, indent=2) + "\n"
                )
                f_out.flush()

                # Save Wave and Tokens
                save_start = time.perf_counter()
                os.makedirs(os.path.dirname(wav_save_path), exist_ok=True)
                import scipy.io.wavfile as wavfile

                speech_np = tts_speech.to("cpu").numpy().squeeze()
                wavfile.write(wav_save_path, sample_rate, speech_np)
                save_elapsed = _elapsed_seconds(save_start)

                audio_seconds = tts_speech.shape[-1] / sample_rate
                total_elapsed = _elapsed_seconds(item_start)
                rtf = total_elapsed / audio_seconds if audio_seconds > 0 else float("inf")
                logging.info(
                    f"[jsonl_generate] item={item_idx}/{len(item_list)} uttid={uttid} done prompt_prep={prompt_prepare_elapsed:.3f}s generate={generate_elapsed:.3f}s save={save_elapsed:.3f}s total={total_elapsed:.3f}s audio={audio_seconds:.3f}s rtf={rtf:.4f} wav={wav_save_path}"
                )
            except Exception as e:
                logging.error(f"Error processing {item.get('uttid', 'unknown')}: {e}")
                import traceback

                traceback.print_exc()

def load_models(
    use_phoneme=False,
    sample_rate=24000,
    llm_dtype="bf16",
    llm_backend="hf",
    hf_attn_implementation=None,
    hf_graph_decode=True,
    hf_graph_max_cache_len=1536,
    hf_graph_warmup_iters=2,
    hf_graph_buckets=DEFAULT_HF_GRAPH_BUCKETS,
    hf_graph_prebuild_buckets=True,
):
    load_models_start = time.perf_counter()

    # Load Speech Tokenizer
    speech_tokenizer_path = os.path.join("ckpt", "speech_tokenizer")
    speech_tokenizer_start = time.perf_counter()
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(
        speech_tokenizer_path
    )
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)
    logging.info(
        f"[load_models] speech_tokenizer={_elapsed_seconds(speech_tokenizer_start):.3f}s"
    )

    # Load Frontends
    frontend_start = time.perf_counter()
    frontend, text_frontend = load_frontends(
        speech_tokenizer, sample_rate=sample_rate, use_phoneme=use_phoneme
    )
    logging.info(f"[load_models] frontends={_elapsed_seconds(frontend_start):.3f}s")

    llama_path = os.path.abspath(os.path.join("ckpt", "llm"))
    tokenizer_path = os.path.abspath(os.path.join("ckpt", "vq32k-phoneme-tokenizer"))

    llm_start = time.perf_counter()
    llm = GLMTTS(llama_cfg_path=os.path.join(llama_path, "config.json"), mode="PRETRAIN")
    llm.inference_backend = llm_backend
    llm.hf_graph_decode = bool(hf_graph_decode)
    llm.hf_graph_buckets = _parse_hf_graph_buckets(
        hf_graph_buckets, hf_graph_max_cache_len
    )
    llm.hf_graph_max_cache_len = max(llm.hf_graph_buckets)
    llm.hf_graph_warmup_iters = int(hf_graph_warmup_iters)
    llm.hf_graph_prebuild_buckets = bool(hf_graph_prebuild_buckets)
    llm.hf_graph_runners = {}
    llm.hf_graph_runner = None

    if llm_backend == "vllm":
        logging.info("Initializing vLLM Engine")
        logging.info(f"Model: {llama_path}")
        logging.info(f"Tokenizer: {tokenizer_path}")
        llm.vllm_engine = LLM(
            model=llama_path,
            tokenizer=tokenizer_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.6,
            enforce_eager=False,
            max_model_len=1024,
            block_size=128,
            disable_log_stats=True,
            logits_processors=[VllmRasLogitsProcessor],
        )
        llm.llama = None
        logging.info("vLLM Engine is Ready!")
    elif llm_backend == "hf":
        if llm.hf_graph_decode and hf_attn_implementation is None:
            hf_attn_implementation = "eager"
            logging.info(
                "[hf_graph] auto-set hf_attn_implementation=eager for static decode graph (may be overridden by npu_fused)"
            )
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        target_dtype = dtype_map.get(llm_dtype, torch.bfloat16)
        logging.info("Initializing HF stepwise LLM")
        logging.info(f"Model: {llama_path}")
        llm.llama = LlamaForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)
        if hf_attn_implementation:
            llm.llama.config._attn_implementation = hf_attn_implementation
            for layer in llm.llama.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = hf_attn_implementation
        llm.llama.eval()
        llm.llama_embedding = llm.llama.model.embed_tokens
        llm.vllm_engine = None
        apply_all_npu_optimizations(llm)
        logging.info(
            "HF stepwise LLM is Ready! attn_implementation=%s",
            getattr(llm.llama.config, "_attn_implementation", None),
        )
    else:
        raise ValueError(f"Unsupported llm_backend: {llm_backend}")

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)
    _maybe_prebuild_hf_graph_runners(llm)

    flow_ckpt = os.path.join("ckpt", "flow", "flow.pt")
    flow_config = os.path.join("ckpt", "flow", "config.yaml")
    flow_start = time.perf_counter()
    flow = yaml_util.load_flow_model(flow_ckpt, flow_config, DEVICE)
    logging.info(f"[load_models] flow={_elapsed_seconds(flow_start):.3f}s")

    token2wav_start = time.perf_counter()
    token2wav = tts_model_util.Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)
    logging.info(
        f"[load_models] token2wav={_elapsed_seconds(token2wav_start):.3f}s total={_elapsed_seconds(load_models_start):.3f}s"
    )

    return frontend, text_frontend, speech_tokenizer, llm, token2wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-TTS Inference Script (Pretrain Mode Only)"
    )
    parser.add_argument("--data", default="example_zh", type=str)
    parser.add_argument("--exp_name", default="_test", type=str)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--use_phoneme", action="store_true", default=False)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument(
        "--llm_dtype",
        default="bf16",
        choices=["fp32", "bf16", "fp16", "int8"],
        help="LLM dtype",
    )
    parser.add_argument("--llm_backend", default="hf", choices=["vllm", "hf"])
    parser.add_argument("--hf_attn_implementation", default=None, choices=["eager", "sdpa"])
    parser.add_argument(
        "--hf_graph_decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hf_graph_max_cache_len", type=int, default=1536)
    parser.add_argument("--hf_graph_warmup_iters", type=int, default=2)
    parser.add_argument("--hf_graph_buckets", type=str, default=DEFAULT_HF_GRAPH_BUCKETS)
    parser.add_argument(
        "--hf_graph_prebuild_buckets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample_method", default="ras", choices=["ras", "topk", "nucleus"])
    parser.add_argument("--sampling_top_p", type=float, default=HF_SAMPLE_TOP_P)
    parser.add_argument("--sampling_top_k", type=int, default=HF_SAMPLE_TOP_K)
    parser.add_argument("--sampling_temperature", type=float, default=HF_SAMPLE_TEMPERATURE)
    parser.add_argument("--dump_tokens", action="store_true", default=False)
    parser.add_argument("--token_dump_dir", default=None, type=str)

    args = parser.parse_args()

    set_npu_jit(False)
    os.environ["ACL_OP_COMPILER_CACHE_MODE"] = "enable"
    cache_dir = os.path.join(CURRENT_DIR, "npu_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["ACL_OP_COMPILER_CACHE_DIR"] = cache_dir

    main_start = time.perf_counter()
    logging.info(
        f"[main] start data={args.data} exp_name={args.exp_name} sample_rate={args.sample_rate} "
        f"use_cache={args.use_cache} use_phoneme={args.use_phoneme} device={DEVICE} "
        f"llm_backend={args.llm_backend} hf_graph_decode={args.hf_graph_decode} "
        f"hf_graph_buckets={args.hf_graph_buckets} sample_method={args.sample_method}"
    )
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=args.use_phoneme,
        sample_rate=args.sample_rate,
        llm_dtype=args.llm_dtype,
        llm_backend=args.llm_backend,
        hf_attn_implementation=args.hf_attn_implementation,
        hf_graph_decode=args.hf_graph_decode,
        hf_graph_max_cache_len=args.hf_graph_max_cache_len,
        hf_graph_warmup_iters=args.hf_graph_warmup_iters,
        hf_graph_buckets=args.hf_graph_buckets,
        hf_graph_prebuild_buckets=args.hf_graph_prebuild_buckets,
    )

    # Create Output Directory
    folder_path = os.path.join(
        CURRENT_DIR, "outputs", f"pretrain{args.exp_name}", args.data
    )
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Output folder: {folder_path}")

    # Run Inference
    jsonl_generate(
        args.data,
        folder_path,
        sample_rate=args.sample_rate,
        use_cache=args.use_cache,
        use_phoneme=args.use_phoneme,
        sample_method=args.sample_method,
        sampling_top_p=args.sampling_top_p,
        sampling_top_k=args.sampling_top_k,
        sampling_temperature=args.sampling_temperature,
        dump_tokens=args.dump_tokens,
        token_dump_dir=args.token_dump_dir,
    )
    logging.info(f"[main] finished total={_elapsed_seconds(main_start):.3f}s")
