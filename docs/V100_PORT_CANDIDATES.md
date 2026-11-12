# V100 (sm_70) 移植清单（已按本仓库逻辑审查）

对比基线：

- **参考源**：`jackieveev/Strata-V100x2`（jmnargi/Strata-V100 fork 线），`0.1.45.7`，本机 `~/strata-v100x2-src`
- **目标**：本仓库 `0.1.59.1`
- **原则**：默认只开 **不改变 token** 的优化；允许「极小的数值差异换明显速度」的开关，但必须单独 opt-in 并写明证据与代价

**审查结论**：`0.1.59.1` 已有 `STRATA_EXPERIMENTAL_SM60`（Volta 能编），缺的是 **Volta 默认开关表**；fork 的多数 prefill 调优在 0.1.59.1 里已被上游重写替换，不能机械搬运（见 P1 一节）。

---

## P0-1 已落地的 sm_70 默认表（`src/core/nv_defaults.cpp`）

11 个开关，每一条都有本仓库内的 bitwise / parity 依据：

| 开关 | 本仓库读取点 | 等价性证据 |
|---|---|---|
| `STRATA_SM70_TABLE` | `fused_gr.cu:2654`、`iq_kernels.cu:6186`、`native_mmvq.cu:3930` | `fused_gr` 注释写 bitwise；expert mode 8 在 P100/P40/Turing 注释里均为 bitwise，Volta 标「待 V100 确认」→ 用 parity 测试补确认 |
| `STRATA_DEQUANT_WIDE` | `dequant_bf16.cu:287` | "the same expression on the same operands … bitwise the same" |
| `STRATA_SIDE_MIXER=2` | `verify.cpp:263` | "bitwise the same outputs" |
| `STRATA_GR_UP75` | `fused_gr.cu:562` | "the same bits" |
| `STRATA_B6_MMVQ_ROWS` | `b6_mmvq_bench.cpp` | "B's output against A's, bit for bit (memcmp)" |
| `STRATA_GU_SWIGLU` | `iq_kernels.cu:4249` | "Every value is the separate kernels'" |
| `STRATA_QSA_ROWS` | `qsa_rows_bench.cpp` | "scores and ids bit for bit" |
| `STRATA_ATTN_DAILY` | `qsa_decode_attn.cu:499` | daily parity harness 逐位比较 |
| `STRATA_GR_NORM_FAST` | `fused_gr.cu:1522` | "rs and xn are bitwise the same" |
| `STRATA_QIL` | `verify.cpp:1631` | "the same bytes"（Turing 默认；V100 由 fork 实测） |
| `STRATA_EXPDEC` | `iq_kernels.cu:2461` | "the output is bitwise the same (expdec_bench checks it)" |

- 总开关：`STRATA_SM70_PERF=0` 整表关闭；用户显式设置的变量永远优先。
- `STRATA_DEFAULTS=026`（0.2.6 之前的总开关）会整表关闭 —— 这张表是本仓库新增的，与 `sm_121` 表同样处理。

### 刻意不放进默认表的 4 个

| 开关 | 原因 | 去向 |
|---|---|---|
| `STRATA_PF_WMMA2` | 只在 `STRATA_PF_WMMA=1` 内生效，单独设是死开关 | 档 B（与 `PF_WMMA` 成对） |
| `STRATA_GR_DOWN_TG=1` | `down_tg()` 在 Volta 会被强制成「1 token/block」，正是 `fused_gr.cu` 基准里**每个 T 都更慢**的那档 | 丢弃 |
| `STRATA_ADAPT_OVERLAP` | fork 自己的注释：隐式拷贝预算会改变缓存驻留，**不保证 token 一致** | 档 B |
| `STRATA_ARGMAX_ROWS_SM70` | 贪心 pick 的另一 kernel；只有 logits 完全并列时才可能不同，但两 kernel 的 tie-break 顺序没有文档 | 档 B |

---

## 档 B：允许「极小质量差异换速度」的开关（脚本里 opt-in）

启动脚本 `/data/models/strata-chenjx-q4` 显式打开：

| 开关 | 收益（据 fork 文档） | 质量代价 |
|---|---|---|
| `STRATA_PF_WMMA=1` + `STRATA_PF_WMMA2=1` | prompt 专家走 FP16 tensor core，对 MMQ `dp4a` **+10% prompt** | FP16 激活代替 q8_1，输出会**略有差异**（非 bitwise） |
| `STRATA_SPLIT_OWN=1` | split 首卡自带 prompt buffer；同批数据 decode 120.2→126.4 tok/s（+5.2%）、scratch 4.10→3.62 GiB | 改变常驻专家集合 → 舍入略变 |
| `STRATA_ADAPT_OVERLAP=1` | 拷贝与下一个 verify 窗口重叠 | 缓存驻留变化，不保证 token 一致 |
| `STRATA_ARGMAX_ROWS_SM70=1` | 单 block 贪心 pick（省一次 kernel） | 仅并列 logits 的 tie-break 可能不同 |

关掉档 B 回到纯 bitwise：

```sh
STRATA_PF_WMMA=0 STRATA_ADAPT_OVERLAP=0 STRATA_SPLIT_OWN=0 STRATA_ARGMAX_ROWS_SM70=0 /data/models/strata-chenjx-q4
```

---

## P1 审查结论：不能搬，已被上游重写

fork 里 P1 那批开关在本仓库的引用数：

| fork 开关 | 本仓库引用 |
|---|---|
| `STRATA_PREFILL_LOAN_EXACT` / `PREFILL_PROMOTE` / `PREFILL_REFILL_OVERLAP` / `PREFILL_HELP_MIN` / `PF_SKIP_UNROUTED` | **0** |
| `STRATA_GDN_REC_FAST` / `EXPERT_PAIR` / `VERIFY_ECAP` / `INPUT_ANCHOR_DEFER` / `POOL_SPIN_ZERO` | **0** |

原因：本仓库的 prefill 已换成 `include/strata/prefill/{routed_stream,share_rules,moe_fused,moe_ws,moe_mmq,moe_fused_iq}.hpp` 一套，fork 的 ring/loan/promote 机制不存在了。搬这些等于在新架构上重写，而且没有 V100 专属收益证据。

**等价物**：`STRATA_SPLIT_OWN_PRIMARY` 的意图（首卡保留 prompt buffer）在目标里由已有的 `STRATA_SPLIT_OWN` / `STRATA_SPLIT_OWN_PLACE` / `STRATA_SPLIT_OWN_BUFFERS` 覆盖，直接用 `STRATA_SPLIT_OWN=1` 即可，不需要新增代码。

---

## P2 / P3

- `STRATA_PREFILL_FP16TC`（`moe_fp16tc.cu`，243+81 行）：fork 里按专家分配 GU/H 容量的紧凑 walk。本仓库 prefill 结构已变，且需要 native Q2_0 pack；先不做。
- `kv_tq` / `STRATA_PROMPT_ATTN_TQ`：本仓库 0 引用，KV 层已重写；只有在要读 trellis KV 格式时才值得。
- `qsa_select.cu` 整文件：本仓库已有 33 处 `qsa_select` 引用与 `qsa_pattn.cu`，属上游已覆盖。
- `STRATA_Q2_0`：只对 Q2_0 权重有意义（本机主力 IQ3_S / IQ4_XS）。

---

## 验证口径（不降质）

1. **parity 测试**（`-DSTRATA_BUILD_TESTS=ON` 构建）：`native_expert_parity`、`mmvq_il_parity`、`gr_parity`、`qsa_merge_daily_parity`、`b6_mmvq_bench`、`qsa_rows_bench`、`expdec_bench` —— 全部要求 memcmp 相同。
2. **端到端 token 对比**：同一 prompt、`temperature 0`、`max_tokens` 固定，分别在 `STRATA_SM70_PERF=0` 与 `=1` 下各跑一次，逐 token 比对；档 B 的开关单独再比一轮，差异只应出现在浮点舍入可解释的位置。
3. **速度**：每轮重启引擎，取中位数，至少 3 个 prompt 长度（2K / 8K / 32K）。
