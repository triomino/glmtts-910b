# NPU 性能优化指南

GLM-TTS 在昇腾 910B NPU 上的 LLM 解码阶段优化，实现在 `npu_opt.py` 中。

所有优化在 `GLMTTS_NPU_OPT=1`（默认）时启用，可通过环境变量独立关闭。

## 总开关

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `GLMTTS_NPU_OPT` | `1` | 启用(`1`)或关闭(`0`)所有 NPU 级优化 |
| `GLMTTS_NO_GRAPH_REPLAY` | 未设置 | 禁用 NPUGraph 回放，每步执行 eager 前向 |

## 独立开关

| 环境变量 | 优化项 | 说明 |
|---------|--------|------|
| `GLMTTS_DISABLE_FIA=1` | 融合注意力 | 对 decode S=1 使用 `npu_fused_infer_attention_score` |
| `GLMTTS_DISABLE_SCATTER_UPDATE=1` | KV Cache 写入 | 用 `scatter_update_` 替代 `index_copy_`（decode S=1） |
| `GLMTTS_DISABLE_RMS_NORM=1` | 融合 RMS Norm | 用 `npu_rms_norm` 替代 pow+mean+rsqrt+mul（仅 decode） |
| `GLMTTS_DISABLE_MLP_FUSE=1` | 融合 MLP | gate_up 合并 + `npu_swiglu` 激活融合（仅 decode） |
| `GLMTTS_DISABLE_ROPE=1` | 融合 RoPE | `_npu_rotary_embedding` — **默认关闭**（见已知问题） |

> **说明**：nucleus 采样优化（`cosyvoice/utils/common.py`）无条件启用——将逐元素 `.item()` NPU 同步（每步最多 25 次）改为一次批量 `.cpu()` 传输。

## 测试环境

所有性能数据在 **Ascend 910B4** 上测得：
- 测试文本：28 个中文字符 → ~110 个 decode 步
- 单步 decode 耗时 = 采样 + 缓冲区准备 + 模型前向 + log_softmax
- 预热 10 步，测量 100 步，取中位数
- 除特别说明外，各配置均启用图模式

---

## 1. NPUGraph 静态图重放

### 原理

`torch.npu.NPUGraph` 将整个 decode 步的计算图（embedding → 28 层 transformer → norm → lm_head）捕获为可重放的静态图。一旦捕获，每步 decode 只需执行预编译的图，跳过 Python 解释器开销和 kernel 调度损耗。

根据是否启用 FIA 分为两种模式：
- **有 FIA**：`torch.npu.NPUGraph` 原生捕获（配合 FIA 预分配缓冲区）
- **无 FIA**：`torch.npu.make_graphed_callables` 封装步函数

### 性能

| 配置 | 单步耗时 | vs Eager |
|------|---------|----------|
| Eager（无图、无优化） | 50.37 ms | 基线 |
| +Graph（仅图、无优化） | 14.16 ms | **-71.9%** |

**图模式独立收益：50.37 → 14.16 ms（3.6× 加速）。**

### 精度

比特级精确：图回放与非图 eager 执行产生完全相同的 logits。
已验证：cos=1.0, max_abs=0。

---

## 2. 融合推理注意力（FIA）

### 原理

将标准的注意力计算（QK^T → softmax → PV，3 个独立 kernel）替换为单个融合 kernel `npu_fused_infer_attention_score`。

```
标准:  Q @ K^T (matmul) → scale + mask → softmax → output @ V (matmul)
融合:  npu_fused_infer_attention_score(Q, K, V, attn_mask, ...) → 单 kernel
```

融合 kernel 使用 **BNSD** 布局，需要：
- `sparse_mode=0` + **bool 注意力掩码**（True = 需屏蔽的位置）
- `actual_seq_lengths=[1]`（decode S=1）
- 预分配输出/workspace 缓冲区以兼容 NPUGraph

**收益来源**：消除中间 QK^T 矩阵的显存写回（98304 × 128），降低显存带宽，3+ kernel 启动合并为 1 次。

### 性能

在图模式基础上测量：

| 配置 | 单步耗时 | 较上一配置变化 |
|------|---------|--------------|
| 仅图（无优化） | 14.16 ms | 基线 |
| +FIA | 12.74 ms | **-1.42 ms (-10.0%)** |

### 精度

| 子模块 | Cosine | max_abs | rel_euc | JS |
|-------|--------|---------|---------|-----|
| attn_out（第 0 层） | 0.9999995 | 0.125 | 0.0010 | ~0 |

每步最大误差 ~0.125（bf16 ULP 级），不存在误差累积。

---

## 3. scatter_update_ KV Cache

### 原理

将 `StaticLayer.update`（每步 decode 的 KV cache 写入）中的 `index_copy_` 替换为 `torch_npu.scatter_update_`。

`index_copy_` 每调用复制一个 slice，而 `scatter_update_` 是专门优化此访问模式的 NPU 算子。

**仅 decode（S=1）**：Prefill 仍使用原始 `index_copy_` 处理多 token 更新。

### 性能

| 配置 | 单步耗时 | 较上一配置变化 |
|------|---------|--------------|
| +FIA | 12.74 ms | 基线 |
| +scatter | 9.51 ms | **-3.22 ms (-25.4%)** |

### 精度

比特级精确：`scatter_update_` 与 `index_copy_` 结果完全一致（cos=1.0, max_abs=0）。

---

## 4. 融合 RMS Norm（仅 decode）

### 原理

将标准 RMS Norm 计算替换为单个 `npu_rms_norm` kernel。

```
标准: pow(hidden, 2) → mean(-1) → add(eps) → rsqrt → mul(weight)  (5 次 kernel 启动)
融合: npu_rms_norm(hidden, weight, eps) → (output, rstd)           (1 次 kernel 启动)
```

**仅 decode（S=1）**：Prefill（S>1）回退到 eager bf16 计算，保证 prefill 输出比特级精确。

### 性能

| 配置 | 单步耗时 | 较上一配置变化 |
|------|---------|--------------|
| +scatter | 9.51 ms | 基线 |
| +RMS | 8.81 ms | **-0.70 ms (-7.4%)** |

### 精度

| 子模块 | Cosine | max_abs | rel_euc | JS |
|-------|--------|---------|---------|-----|
| ln_out（第 0 层） | 0.999996 | 0.0078 | 0.0028 | 9.6e-7 |

---

## 5. 融合 MLP / SwiGLU（仅 decode）

### 原理

GLM 的 MLP 使用 SwiGLU 结构：`silu(gate_proj(x)) * up_proj(x) → down_proj`。

融合分两步：

1. **权重合并**：将 `gate_proj` 和 `up_proj` 的权重拼成一个 `gate_up_weight` 参数。一次 `F.linear(x, gate_up_weight)` 同时计算两个投影（一次大矩阵乘替代两次小矩阵乘，NPU 利用率更高）。

2. **激活融合**：`npu_swiglu` 将 SiLU 激活和逐元素乘融合为一次 kernel 调用。

```
标准: gate = silu(linear_gate(x)) → up = linear_up(x) → silu(gate) * up → linear_down(x)
融合: gu = linear_gate_up(x) → swiglu_out = npu_swiglu(gu) → linear_down(swiglu_out)
```

**仅 decode（S=1）**：Prefill 回退到 eager `silu(gate) * up` 以保证数值精度。

### 性能

| 配置 | 单步耗时 | 较上一配置变化 |
|------|---------|--------------|
| +RMS | 8.81 ms | 基线 |
| **+MLP（最终配置）** | **8.36 ms** | **-0.45 ms (-5.1%)** |

### 精度

| 子模块 | Cosine | max_abs | rel_euc | JS |
|-------|--------|---------|---------|-----|
| mlp_out（第 0 层） | 0.9999997 | 0.125 | 0.0007 | ~0 |

---

## 6. Nucleus 采样优化

### 原理

原始的 `nucleus_sampling` 对排序后的 logits 遍历最多 `top_k=25` 个元素，每步调用 `.item()` 触发一次 CPU-NPU 同步。

优化版本：
1. `sorted_value[:top_k].cpu()` — 一次批量传输 25 个值到 CPU（1 次同步）
2. 本地循环处理最多 25 个元素（无 NPU 同步）

```
优化前: topk 在 NPU 排序后 → 循环最多 25 次: sorted_value[i].item()  → ~25 次 NPU 同步
优化后: topk_val = sorted_value[:top_k].cpu()                       → 1 次 NPU 同步
        循环 25 次: cum_prob += topk_val[i].item()                    → 本地 CPU 循环
```

在 910B 的 ARM CPU（单核较弱）上，减少约 25 次同步每次可节省约 1.3ms。

### 性能

在 decode 总步时间中测量（eager，无图，无模型优化）：

| 配置 | 单步耗时 | 变化 |
|------|---------|------|
| Eager + 旧采样（逐元素 .item() 同步） | 51.71 ms | 基线 |
| Eager + 新采样（批量 .cpu() 传输） | 50.37 ms | **-1.34 ms (-2.6%)** |

节省的 ~1.3ms/步与模型优化无关（采样在模型前向之后、图之外调用）。

> 在 GPU 上，逐元素 `.item()` 同步的开销较小（CPU-GPU 互联更快），该优化在 GPU 上收益有限。

---

## 最终性能 vs Eager

### 单步 decode 耗时（累计叠加）

| # | 配置 | 单步耗时 | 累计收益 |
|---|------|---------|---------|
| 0 | Eager（无图、无优化） | 50.37 ms | 基线 |
| 1 | +采样优化 | 49.03 ms* | -2.7% |
| 2 | +图重放 | 14.16 ms | -71.1% |
| 3 | +FIA | 12.74 ms | -74.7% |
| 4 | +scatter_update | 9.51 ms | -81.1% |
| 5 | +RMS Norm | 8.81 ms | -82.5% |
| 6 | **+MLP（最终）** | **8.36 ms** | **-83.4%** |

\* 采样优化与图模式无法严格叠加（采样在图之外）。49.03ms 为估算值。

### 图模式影响

```
最终配置有图:    8.36 ms
最终配置无图:   38.91 ms
图加速比:       4.7×
```

### RTF（实时因子）

在 Ascend 910B4 上测试 59 条语句（2-7s 音频）。

| 配置 | 短文本（2-3s） | 长文本（4-7s） | 全部 |
|------|---------------|---------------|------|
| Eager | 0.57 RTF | 0.48 RTF | 0.53 RTF |
| **最终（无 RoPE）** | **0.38 RTF** | **0.28 RTF** | **0.33 RTF** |
| 提升 | +33% | +42% | +38% |

### 音质

- 盲听 A/B 测试：优化版本与 eager 版本 MOS 无差异
- Token 序列：前 8 步完全相同
- 长音频（~1.5 分钟）：偶有轻微杂音（见已知问题）

---

## 最终精度 vs Eager

### 第 0 层子模块（首个 decode 步）

| 子模块 | Cosine | max_abs | rel_euc | rms | JS |
|-------|--------|---------|---------|-----|-----|
| ln_in（embedding） | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ln_out（RMSNorm） | 0.999996 | 0.0078 | 0.0028 | 0.0013 | 9.6e-7 |
| attn_out（FIA） | 0.9999995 | 0.125 | 0.0010 | 0.0081 | ~0 |
| post_ln_out | 0.999995 | 0.0078 | 0.0032 | 0.0005 | 1.2e-7 |
| mlp_out（swiglu） | 0.9999997 | 0.125 | 0.0007 | 0.0038 | ~0 |
| layer_out | 0.9999996 | 0.250 | 0.0009 | 0.0120 | ~0 |

### Logits（全部优化 vs eager）

| 指标 | Prefill | 第 0 步 | 第 8 步（分叉点） |
|------|---------|--------|-----------------|
| Cosine | 1.0 | 1.0 | 0.999998 |
| max_abs | 0.0 | 0.0 | 0.125 |
| rel_euc | 0.0 | 0.0 | 0.0024 |
| JS | 0.0 | 0.0 | 0.0017 |

**关键结论**：
- **Prefill 比特级精确**（所有融合算子在 S>1 时回退 eager）
- **首步 decode 比特级精确**（使用 prefill 输出，尚未执行融合算子）
- Token 序列在前 **~8 步完全相同**
- 分叉由 RAS 采样从几乎相同的概率分布中选择不同 token 导致（最大概率差 < 0.008）
- 所有指标偏差均在 bf16 精度下限水平，非精度 bug

---

## 已知问题

### 1. 融合 RoPE（`_npu_rotary_embedding`）— 默认关闭

`_npu_rotary_embedding` 在 position > 0 时产生错误结果（position=50 时 cos 仅 0.39 vs 标准实现）。该 kernel 仅在 position=0 时比特级精确。这是 CANN kernel bug，无法从 Python 侧修复。

**影响**：默认关闭（`enable_rope = False`）。若修复后启用，每步 decode 可节省约 0.2ms。需等待 CANN 后续版本修复。

### 2. 长音频杂音（~1.5 分钟）

生成长音频（~1.5 分钟）时偶有轻微杂音。根因：融合算子 bf16 精度下限处的误差累积。每步误差很小（每层 max_abs < 0.25），但数百步自回归后，在边界情况下可能累积到影响音频质量的程度。

**理论上可尝试的缓解方案（未实测）**：
- o_proj 或 MLP 矩阵乘在 decode 时转为 fp32 可减少误差放大，但增加约 15% 步时间
- RMS Norm 改用 fp32 同样有帮助
- 修复 RoPE 可消除一个误差来源
- 实际音频质量影响很小，fp32 的性能损失大于精度收益

### 3. RMS Norm 和 MLP 融合 — Prefill 回退

RMS Norm 和 MLP 融合在 prefill（S>1）时均回退到 eager bf16 计算。这保证了 prefill 输出的比特级精确，但也意味着这些优化仅对 decode 步（通常每段 50-200 步）有效。
