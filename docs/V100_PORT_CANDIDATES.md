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

---

## 本机实测（2026-11-12，2×V100-SXM2-16G，IQ4_XS，split 25，262144 ctx）

同一组固定 prompt（约 2K / 8K / 32K token，T=0，每次唯一 nonce 避开前缀缓存）：

| 配置 | prefill tok/s (2K / 8K / 32K) | decode tok/s (2K / 8K / 32K) |
|---|---|---|
| 官方 0.1.59（基线，同 config） | 812 / 1423 / 2044 | 44.4 / 54.5 / 57.7 |
| 本仓库位 A（sm_70 表开，档 B 全关） | 821 / 1444 / 2066 | 42.1 / 55.3 / 62.2 |
| 位 A + `PF_WMMA=1` | 836 / 1466 / 2097 | 40.7 / 63.1 / 80.4 |
| 位 A + `SPLIT_OWN=1` | **734 / 1066 / 1154** | 37.8 / 52.6 / 60.0 |
| 位 A 关（`STRATA_SM70_PERF=0`） | 820 / 1426 / 2045 | 39.1 / 53.8 / 62.5 |

结论：

- **sm_70 表在这台机器上只有 ~1%**（fork 在 2×V100-32G、IQ3_S、split 16 上测到 +11.4%）。bitwise 成立、无质量风险，但不要指望它带来明显提速。
- **`STRATA_SPLIT_OWN=1` 在 16 GB 卡上是负收益**（prefill −26~−44%）：它给每个 stage 留 ~1904 MiB prompt buffer，专家缓存从 3612 槽掉到 3056 槽，而本机无 NVLink，缓存未命中的代价比省下的拷贝更贵。默认关。
- **`PF_WMMA=1` 与基线持平**（prefill +1~2%，落在噪声内）：WMMA tile 只接 down 为 Q2_0 / IQ4_NL 的层（`prefill.cpp` 的 `wmma_l`），这份 UD-IQ4_XS 只有 16 / 48 层符合；另外 30 层的 Q5_0 down 当时连 MMQ 都没进（见下节）。它改变数值，故默认关。
- 本仓库与官方引擎持平（prefill +1%，32K decode 略高但样本小）——**换引擎本身没有变快，价值在于代码已跟上 0.1.59.1 并带上了 sm_70 表**。

启动脚本 `/data/models/strata-chenjx-q4` 的默认值按此定版：档 A 开、档 B 全关，每个档 B 开关仍可用环境变量单独打开做实验。

---

## all-cards split engine（paskaa 路线）：硬件可行，**模型不支持**

- 构建：`-DSTRATA_BUILD_SPLIT=ON -DSTRATA_SPLIT_CARDS=2 -DSTRATA_SPLIT_ARCH=70` 在本机通过，产物 `build-split/src/split/strata-split-2`（19 MB）。
- 试跑（131072 ctx，`FNX_EXPERT_RAM_GIB=3 FNX_KV8=0`，两卡空闲）：

```
gguf: 216 dense/small tensors converted to Q8_0/F32 at load (the file's routed experts keep their types)
ERR the all-cards split engine does not read this model's routed experts
    (blk.0.ffn_down_exps.weight is ggml type 6); use the normal layer split
```

- 结论：**这条路被模型格式卡住，不是被 16 GB 显存卡住**。当前 `UD-IQ4_XS-abliterated` 的 routed experts 是 type 6，sfnx 不支持；paskaa 的 +57% 是在 Q4_K_XL / QAD NVFP4 权重上测的。要走这条路需要换一份 sfnx 支持的量化权重（或重新量化），本机现有权重不行。
- 因此这条路线在"不换模型"的前提下**关闭**，不再投入。

---

## Q5_0 down 进 MMQ（`STRATA_ORCA_Q4KS_MMQ` 改为默认 ON）

**问题**：`native_experts.txt` 显示这份 UD-IQ4_XS 的 down 专家有 30 / 48 层是 Q5_0（type 6），其余为 IQ4_NL（20）/ Q2_0（42）；gate/up 是 IQ3_XXS / IQ2_S / IQ4_XS。
CUDA 构建的 MMQ 类型表（`cmake/strata_ggml.cmake` 的 `_strata_mmq_types`）**不含 q5_0**，只有 `STRATA_ORCA_Q4KS_MMQ=ON` 才加。
`mmq_plan()` 要求一层的 gate/up 与 down 都有 MMQ tile，否则整层回落 FP16 路径：每个专家先反量化成 FP16 写回显存，再各调一次 cuBLAS。
`STRATA_PREFILL_TIMING=1` 里这部分记在 `dequant`（17–23%）和 `gemm`。

**改动**：`CMakeLists.txt` 把该选项默认值改为 ON（只多编一个 Q5_0 MMQ 实例；每卡少 2 个专家缓存槽，4385→4383、3512→3511）。

**实测**（2026-11-12，2×V100-SXM2-16G，split 25，262144 ctx，T=0，唯一 nonce，512 token 输出；`~/tune/bench.py`）：

| | 8192-token 块的首卡 GPU 时间 | `dequant` 段 | prompt tok/s 14K | prompt tok/s 61K |
|---|---|---|---|---|
| 旧构建 | 3968 ms | 744 ms | 1364 / 1388 / 1381 / 1279 | 1960 / 1959 |
| Q5_0 MMQ | 2745 ms（−31%） | 103 ms | 1424 / 1483 / 1489 / 1473 | 2094 / 2086 / 2098 |

- 3.2K（单块）prompt 不变（≈780–870 tok/s，噪声内）：单块时两张卡串行，且首卡 `wait copy` 从 93 ms 升到 356–460 ms，瓶颈变成 PCIe 流式拷贝。
- 长 prompt 稳态的关键路径是末卡：GPU ≈3.3 s/块 + 草稿层 ≈0.3 s/块；`gemm` ≈39%，`qsa select+attn` 随上下文涨到 ≈23%。
- decode 不受影响（MMQ 只在 prompt 路径）。
- 数值：这 30 层的激活从 FP16 改为 MMQ 的 q8_1（与其余 18 层、以及 llama.cpp 默认一致）。质量回归：182 工具、22.4K token 的 agent 请求两边都返回同一个结构化 `read_file(src/main.cpp)`；4 道短题答案一致（其中一道措辞不同，同一构建重启两次也会这样，因为 `STRATA_PREFILL_CPU_SHARE=auto` 按计时分配）。

## Q5_0 down 进 WMMA W2 tile（`STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1`，opt-in）

**改动**：`gemm_iq_f16_w2_kernel`（`src/kernels/cuda/iq_kernels.cu`）加 Q5_0 traits 与反量化（4 块 / 88 B 一级，`(1024+q) − 1040` 得精确整数，再与 d 相乘只舍入一次，与 `iq_dequant_f16` 逐位一致）；`prefill.cpp` 的 `wmma_l` 在 W2 开启时接受 Q5_0 down（`gemm_iq_f16_grouped_q5_0()`）。`pf_wmma_parity` 在 `STRATA_PF_WMMA2=1` 下多测一项 Q5_0 down（两卡 rel 2e-6，通过）。

**实测**（同上口径，`STRATA_PREFILL_TIMING=1`，8192 块 15 个取均值）：

| | gemm down | gemm gate/up | 首卡 GPU 时间 | prompt 14K | prompt 61K ×2 |
|---|---|---|---|---|---|
| 生产（MMQ） | 507 ms | 805 | 3088 | 1445 | 2062 / 1998 |
| W2（仅 IQ4_NL 16 层，旧二进制） | 440 | 799 | 3045 | 1453 | 2078 / 2024 |
| W2 + Q5_0（30 层） | **273** | 785 | **2935** | **1515** | **2134 / 2078** |

- down 段 −46%，prompt +3.5~5%；3.2K 单块不变（PCIe 流式拷贝主导）。decode 不受影响。
- 数值：这两层产品的激活从 q8_1 变 FP16（精度更高，但结果与默认不逐位一致），所以保持 opt-in。质量回归：agent 请求同一个 `read_file(src/main.cpp)`，4 道短题与生产一致。

## decode：CPU 专家池

nsys：decode 时 GPU 20% 时间在 `wait_flag_ge_kernel`。`perf record`（3.2K prompt、decode 中 10 s）看 CPU：`ExpertPool::worker` 自旋 49%；算力部分 ggml 单 token dot 占大头（`iq3_xxs` 14.7%、`q5_0` 8.7%、`iq4_xs` 6.7%），多 token 内核合计不到 6%。即**大多数缺失专家一次只有 1 个 token**，且池子一半时间在等，瓶颈是逐层同步延迟而非吞吐。

- **`--pcie-frac` 扫描**（同一二进制，1024 token 输出，3.2K×3 + 14K，tok/s）：0.00 ≈79、0.05 ≈75（含一次 67 的离群）、0.10 ≈78、**0.20 ≈80（现值）**、0.35 ≈72、0.50 ≈63、去掉参数走 auto（逐 stage 探测 12.1 GB/s，PcieAuto 收敛到 0.15）≈79。0.20 已在平台上，往上明显变慢，**不改**。
- **AVX2 多 token Q5_0 down**（`q50_256_down_rows`，`src/kernels/cpu/iq_avx2.cpp`）：这台 E5-2686 v4 没有 AVX-512，30 层 Q5_0 down 原来只能逐 token 调 ggml 的 dot（512-bit 版要 VNNI）。新内核每块权重只解一次，逐 token 运算与 ggml AVX2 dot 完全同序，首次使用时与链接的 ggml 逐位比对（`q50_256_ok`），不一致就不用；`STRATA_NO_Q50_256=1` 关。`q50_avx2_parity`：1–8 token 全部逐位一致，单线程 2–8 token 1.3–1.5×；`native_expert_parity --synthetic iq4_xs/q5_0`、`iq3_xxs/q5_0`、`q4_K/q5_0` 通过（含 1 vs 3 token 宽度不变性）。
  端到端 decode（同一二进制交替开关，各 2 轮 × 5 请求）：开 78.7 / 79.2，关 79.3 / 80.0，**噪声内无收益**——它只覆盖 nt ≥ 2 的那 ~4% 样本。逐位一致、无风险，保留默认开；要动 decode 得先解决 nt = 1 的单 token 延迟和池子同步。

## 同日测过、没有收益的项（别再测）

| 项 | 结果 |
|---|---|
| PLE 表锁内存（`prlimit --memlock=unlimited`，日志确认 `locked in RAM`） | decode 3×3.2K 中位 80.2 vs 84.7 tok/s，无收益 |
| `STRATA_MTP_STAGE0=1`（草稿层放首卡） | decode 69.8 vs 84.7，**−18%** |
| `STRATA_PIPELINE_THETA=0.35`（需 `STRATA_PIPELINE_DEBUG=1`） | decode 77.9 vs 84.7，变慢 |
| `STRATA_PREFILL_AUTO_MAX=16384` + `STRATA_SPLIT_CHUNK16K=1` | 块变 16384、每卡借 7.31 GiB；61K 2098（不变），14K 变慢 |
| `STRATA_PREFILL_HELP=1` / `STRATA_PREFILL_PIPE=1` | 噪声内 |
| `--kv-resident` 下调 | 未测：32768 cell 在 VRAM 里总共只占 ≈0.2 GiB（1.55 GiB 是 229376 cell 的 RAM 部分），降到 8192 最多多 ≈75 槽（≈1%） |

`STRATA_SPLIT_MTP_BATCH=1`（末卡分批填草稿层的 prompt K/V）：prompt 61K 2149 / 2155 vs ≈2090（+3%），但 decode 14K 均值 72.6（4 次）vs 77.1（7 次）、61K 67.0 vs 69.7，约 −4~6%（草稿变差、接受率降）。只影响草稿、不影响验证后的输出，但 decode 更重要，**不开**。
