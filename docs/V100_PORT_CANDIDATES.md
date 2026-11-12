# V100 (sm_70) 移植候选清单

对比基线：

- **参考源**：`jackieveev/Strata-V100x2`（= jmnargi/Strata-V100 的 V100 fork 线），`project(strata VERSION 0.1.45.7)`，本机路径 `~/strata-v100x2-src`
- **目标**：本仓库，`0.1.59.1`
- **方法**：两边环境变量集合求差、`src/`+`include/` 文件清单求差、`sm_70/volta/V100` 关键字分布、以及 fork 自带 `docs/NVIDIA_V100.md` 的实测数据

**总体判断**：0.1.59.1 已经有 `STRATA_EXPERIMENTAL_SM60` 构建开关（Volta 能编），因此不需要重建 Volta 构建支持。真正缺的是 **Volta 专属默认开关表**（本仓库 `src/core/nv_defaults.cpp` 只有 sm_86 / sm_60 / sm_121，**没有 sm_70 分支**）和几项 V100 内核/调度优化。

---

## 总览：按「收益 / 风险」排序

| # | 项目 | 收益 | 风险 | 改动面 |
|---|---|---|---|---|
| **P0-1** | sm_70 默认开关表（`STRATA_SM70_PERF`） | **高**（fork 实测 2×V100 16G：83.1 vs 74.6 tok/s，**+11.4% decode**） | **低**：纯 env 表，无新代码路径，有总开关 | `src/core/nv_defaults.cpp` 一个分支（约 40 行） |
| **P0-2** | 表内 15 个开关的存在性核对 | —（前置检查） | 低 | 无改动 |
| **P1-1** | `STRATA_SPLIT_OWN_PRIMARY` + `STRATA_SPLIT_OWN_HEADROOM_MIB` | 中（split 首卡保留自己的 prompt buffer，减少每 chunk 借还专家缓存；fork 同批数据：scratch 4.10→3.62 GiB、借出 slot 3374→2976、decode 120.2→126.4 tok/s） | 中低：**opt-in**，改变驻留专家集合 | `src/program/generate.cpp` 约 100 行（钩子点 `no_prefill_borrow`、`prefill_auto`、`VramPlan` 本仓库都有） |
| **P1-2** | prefill 贷款/提升族：`STRATA_PREFILL_LOAN_EXACT`、`STRATA_PREFILL_PROMOTE`、`STRATA_PREFILL_REFILL_OVERLAP`、`STRATA_PREFILL_HELP_MIN`、`STRATA_PF_SKIP_UNROUTED` | 中（prefill 阶段少借 slot、热门专家不被冲掉） | 中：数值/命中率变化，**全部 opt-in 或默认关** | 1–3 文件，多为开关 + 几十行 |
| **P1-3** | 小开关族：`STRATA_POOL_SPIN_ZERO`、`STRATA_INPUT_ANCHOR_DEFER`、`STRATA_GDN_REC_FAST`、`STRATA_EXPERT_PAIR`、`STRATA_VERIFY_ECAP` | 低–中（个别是热点：spin 循环 CPU 时间 139.8→46.2 ms @SPIN_US=0） | 低（多为局部实现替换） | 分散，1–5 文件 |
| **P2-1** | `STRATA_PREFILL_FP16TC` 紧凑 tensor-core prompt 走法（`moe_fp16tc.cu`） | **高**（fork 文档：WMMA prompt experts 对 MMQ dp4a **+10% prompt**；紧凑 walk 进一步按专家而非 token 分配 GU/H 容量） | **高**：新 CUDA 内核 + 几何校验；要求 native pack 且每层 MMQ-eligible、Q2_0 格式；本仓库 prefill 结构已变 | 新 `.cu` 243 行 + `.hpp` 81 行 + `prefill.cpp` 22 处钩子 |
| **P2-2** | `STRATA_PF_WMMA_LEAN`、`STRATA_PF_WMMA2_N64` | 中 | 中：只在 opt-in `STRATA_PF_WMMA=1` 路径内生效 | 1–3 文件 |
| **P3-1** | `kv_tq`（trellis KV）+ `STRATA_PROMPT_ATTN_TQ` | 不确定（取决于是否用 trellis KV 格式） | **高**：本仓库 KV 路径重写过，`kv_tq`/`tq_codebook` 引用数为 **0**，等于重写 | 264+108+53 行 + 约 17 个引用点 |
| **P3-2** | `qsa_select.cu` 整文件（2727 行） | 低（本仓库已有 33 处引用，且新增 `qsa_pattn.cu`，属上游已覆盖） | 高（整文件冲突） | 不建议整文件移植 |
| — | `STRATA_Q2_0`、`STRATA_SM70_CHUNK`、`STRATA_TRACE_RESUME`、`STRATA_TEST_*` | 低 | — | 不建议（Q2_0 只对 Q2_0 权重有意义；本机日常 IQ3_S/IQ4_XS） |

---

## P0-1 sm_70 默认开关表（先做这个）

参考实现：`~/strata-v100x2-src/src/core/nv_defaults.cpp` 的 `mj == 7 && mn == 0` 分支。

fork 实测（该文件注释原文）：2×V100 16 GB，IQ3_S，layer split 16，`--pipeline-windows 2`，A/B/A/B/A/B、每次测量前重启，**83.1 vs 74.6 tok/s 中位数（+11.4%）**；上游 PR #3411 在双 V100 32 GB 上 IQ3_S/MTP4 为 +5.8/+7.6% 与 +4.7/+4.8%，10 条输出逐 token 一致。

要移植的 15 个开关：

```
STRATA_SM70_TABLE=1        Volta 总开关：interleaved dense table、expert mode 8、hidden HC up
STRATA_DEQUANT_WIDE=1      向量化 dequant 写回
STRATA_ADAPT_OVERLAP=1     仅 Linux：拷贝与 verify 窗口重叠
STRATA_PF_WMMA2=1          opt-in PF_WMMA 路径里的第二种 tile
STRATA_SIDE_MIXER=2        ┐
STRATA_GR_UP75=1           │ 上游 PR #3411 的七开关栈
STRATA_B6_MMVQ_ROWS=1      │
STRATA_GU_SWIGLU=1         │
STRATA_QSA_ROWS=1          │
STRATA_ATTN_DAILY=1        │
STRATA_GR_NORM_FAST=1      │
STRATA_ARGMAX_ROWS_SM70=1  │
STRATA_QIL=1               │
STRATA_EXPDEC=1            │
STRATA_GR_DOWN_TG=1        ┘
```

**P0-2 前置核对结果**（本次已在 0.1.59.1 上 grep）：15 个开关名**全部仍然被读到**，但参考数差异说明部分语义已收窄，逐项 A/B 时注意：

| 开关 | 0.1.59.1 文件数 | 参考源文件数 |
|---|---|---|
| `STRATA_SM70_TABLE` | 3 | 5 |
| `STRATA_DEQUANT_WIDE` | 2 | 3 |
| `STRATA_ADAPT_OVERLAP` | 6 | 8 |
| `STRATA_PF_WMMA2` | **1** | 6 |
| `STRATA_GR_NORM_FAST` | **1** | 2 |
| `STRATA_ARGMAX_ROWS_SM70` | **1** | 2 |
| `STRATA_QIL` | **1** | 2 |
| `STRATA_SIDE_MIXER` / `EXPDEC` | 5 / 5 | 4 / 3 |
| 其余（GR_UP75、B6_MMVQ_ROWS、GU_SWIGLU、QSA_ROWS、ATTN_DAILY、GR_DOWN_TG） | 2–4 | 2–3 |

实现要点：

1. 在 `nv_default_env()` 里加 `mj == 7 && mn == 0` 分支，带 `STRATA_SM70_PERF=0` 关闭开关（与 sm_86/sm_60 的表同构，直接照抄 `off()` 逻辑）。
2. `apply_nv_defaults()` 已经存在并被 `generate.cpp` 在启动时调用，无需新接线。
3. 用户显式设置的变量永远优先（现有 `std::getenv(...) != nullptr` 判断已经保证）。
4. 风险点：这是一张"实测更快"的表，不是 token 级恒等保证。某些开关（例如 `ATTN_DAILY`、`ARGMAX_ROWS_SM70`）可能改变舍入 → 加载后先跑 `--exact-pp` 类验证（fork 的 identity gate：10 条 prompt + exact_pp 4K/20K），确认输出一致再看速度。

---

## P1-1 `STRATA_SPLIT_OWN_PRIMARY`（第二优先）

参考实现：`~/strata-v100x2-src/src/program/generate.cpp:6239` 起，注释块 "A SPLIT'S FIRST CARD KEEPS ITS OWN PROMPT BUFFERS"。

- 语义：双卡 layer split 下，第一张卡用自己缓存之外的空闲 VRAM 自带 prompt buffer，而不是每个 chunk 通过 loan 借专家槽位（借一次要逐出专家、再从 PCIe 重读）。
- 开关：`STRATA_SPLIT_OWN_PRIMARY`（opt-in，1 开）、`STRATA_SPLIT_OWN_HEADROOM_MIB`（保留的余量）。
- 目标仓库对应点都在：o.no_prefill_borrow、o.prefill_auto、strata::prefill::Prefill::bytes_needed、VramPlan。
- 验证：加载后对比 `primary cache slots` / `borrowed` 日志行，以及长 prompt 的 prefill 时间。

## P1-2 prefill 贷款/提升族

| 开关 | 作用 | 参考实现位置 |
|---|---|---|
| `STRATA_PREFILL_LOAN_EXACT=1` | 按实际读取长度给 loan 定价，不再向上取整到 256 行 | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_PROMOTE=N` | 归还 loan 时每 stage 最多做 N 次专家替换，防热门专家被冲掉 | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_REFILL_OVERLAP` | 归还与重填重叠 | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_HELP_MIN` | 空闲 stage 开始帮忙的 token 门槛 | `src/prefill/prefill.cpp` |
| `STRATA_PF_SKIP_UNROUTED=1` | 本 chunk routing 没选中的 pinned expert 不拷进 ring（只省 DMA） | `src/prefill/` |

注意：`LOAN_EXACT` 与 `PREFILL_CPU_SHARE_MAX=6144` 是**配套**的（fork 里把 CPU share 上限从 3072 提到 6144 就是靠它避免过度借用）。本机 v100fork 配置现在正是 `CPU_SHARE_MAX=6144`；只搬 `LOAN_EXACT` 而不动上限更稳。

## P2-1 `STRATA_PREFILL_FP16TC`（收益最大、风险也最大）

- 文件：`src/prefill/moe_fp16tc.cu`（243 行）+ `include/strata/prefill/moe_fp16tc.hpp`（81 行）
- 作用：compact tensor-core walk —— GU/H 按**专家**而不是按 token 分配行容量，用 FP16 tensor core 一次算一组专家；`tc_plan()` 在设备/pack 不满足时自动退回 MMQ 参考路径。
- 前置条件（fork 注释）：Volta cc 7.0 + native pack + 每层 MMQ-eligible + 权重是内核支持的 Q2_0 形式。
- 移植成本：`prefill.cpp` 里 `tc_plan`/`fp16tc_` 共 22 处钩子，且本仓库 prefill 已重构过（`fp16_expert` 已抽成 `src/prefill/fp16_expert.hpp`），不能直接贴。
- 建议：**先不做**，除非 P0/P1 落地后仍嫌 prefill 慢；做的时候先把 `moe_fp16tc.cu` 单独编进一个 benchmark（fork 有 `fp16tc` 相关的 parity 测试可以照搬）。

## P3 不建议

- `kv_tq`（trellis KV，264+108+53 行，`tq_codebook`）：本仓库 0 处引用，KV 存储层已重写；只有在确定要读 trellis KV 格式时才值得。
- `qsa_select.cu`（2727 行）：本仓库已有 33 处 `qsa_select` 引用且新增 `qsa_pattn.cu`，说明上游已覆盖；整文件贴入必然冲突。
- `STRATA_Q2_0`：只对 Q2_0 权重有意义，本机主力是 IQ3_S / IQ4_XS。
- `STRATA_TEST_*`、`STRATA_TRACE_RESUME`：测试/诊断用，不属于性能优化。

---

## 建议落地顺序

1. **P0-1 + P0-2**：表 + 逐项 A/B（每项单独关掉测一次，记录 tok/s 与 token 一致性）。预期收益最大、可回滚（`STRATA_SM70_PERF=0` 一键关）。
2. **P1-1**：split 首卡自有 buffer。用双卡 `--pipeline-windows 2` 场景测 prefill 与 decode。
3. **P1-2 / P1-3**：按日志里 `borrowed cache slots`、`expert cache hit rate` 的变化决定留哪些。
4. **P2 视收益**：先补 `docs/NVIDIA_V100.md` 的"what runs on Volta"表，再决定要不要搬内核。

## 验证口径（与参考源一致，便于横向对比）

- 构建：`cmake -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`
- 模型：IQ3_S（参考源的主测量模型），`--layer-split 16`，`--pipeline-windows 2`，`--spec 4`
- 每项测量：同一 prompt、`temperature 0`、每轮重启引擎、取中位数；至少 3 个 prompt 长度（2K / 8K / 32K）
- 正确性：先跑 token 一致性（同 prompt 输出逐 token 比对），再跑速度

> 数据出处：`~/strata-v100x2-src/docs/NVIDIA_V100.md`、`~/strata-v100x2-src/src/core/nv_defaults.cpp` 注释、以及本次对两边源码的 grep/文件差统计。fork 文档中部分数字与 Gyro-M/0.2.11 移植同批，归属需要在移植时用 A/B 复核。
