# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

from typing import List

import torch

IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1, temperature=1.0):
    import os
    _use_dst = os.environ.get("GLMTTS_NPU_OPT", "1").strip().lower() not in ("0", "false")
    _use_dst = _use_dst and os.environ.get("GLMTTS_DISABLE_DST_SAMPLING", "").strip().lower() not in ("1", "true")
    if _use_dst:
        top_ids = dst_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    else:
        top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k, temperature=temperature)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


# Adapted from Ascend ModelZoo CosyVoice2 optimization patch.
# Source: https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/audio/CosyVoice2/800I/diff_CosyVoice_800I.patch
# Replaces the Python for-loop in nucleus_sampling with vectorized PyTorch ops.
def dst_sampling(weighted_scores, top_p=0.8, top_k=25):
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    # NPU bf16 cumsum on long vectors (e.g. 98304) is ~70x slower than float32.
    # This is a known gap: CANN Cumsum does not have an optimized bf16 aicore kernel.
    # Cast to float32 restores normal performance with negligible precision impact.
    sorted_value = sorted_value.float()
    cum_sum = torch.cumsum(sorted_value, dim=0)
    n = sorted_value.size(0)
    device = cum_sum.device
    pre_cum_sum = torch.cat([torch.zeros(1, device=device), cum_sum[:-1]])
    indices = torch.arange(n, device=device)
    condition = (pre_cum_sum < top_p) & (indices < top_k)
    max_i_tensor = torch.where(condition, indices, torch.tensor(-1, device=device))
    n_selected = max_i_tensor.max() + 1
    selected_prob = sorted_value[:n_selected]
    selected_indices = sorted_idx[:n_selected]
    top_ids = selected_indices[selected_prob.multinomial(1, replacement=True)]
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25, temperature=1.0):
    prob, indices = [], []
    cum_prob = 0.0
    scaled_scores = weighted_scores / temperature
    sorted_value, sorted_idx = scaled_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[:, :, :mel_overlap_len] = fade_in_mel[:, :, :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[:, :, -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)
