# 交接文档：Strata V100 提速调优（给下一个 agent）

最后更新：2026-11-15 (Asia/Shanghai)。

**目标：在尽量不降低生成质量的前提下提速。** 用户确认当前生产配置的质量正常，不需要排查质量问题。每项提速改动做完后，用第 6 节的回归集确认质量没有变差即可。

---

## 1. 机器与现状

| 项 | 值 |
|---|---|
| 主机 | `<SERVER-HOST>`，Ubuntu 22.04，**服务器时钟是 UTC**（比北京慢 8 小时） |
| SSH | `<SSH-USER>@<SERVER-IP>:22`，密码 `<SSH-PW>`（sudo 同密码）——**真实值不进仓库，见仓库外本地副本 `V100-TUNING-HANDOFF.md`** |
| GPU | 2 × Tesla V100-SXM2-16GB（sm_70，各 Gen3 x16） |
| GPU 互连 | **PCIe-only：本机硬件就没有 NVLink（用户确认，2026-11-15），不要再假设可激活或作为调优方向**；拓扑 `PHB`（走 CPU 的 PCIe 桥），`cudaDeviceCanAccessPeer` 双向可用 |
| CPU / 内存 | Xeon E5-2686 v4，18 核 ×2 SMT = 36 线程，单 NUMA，125 GB RAM，**有 AVX2，没有 AVX-512 / VNNI** |
| 磁盘 | 单块 256 G NVMe（`/`，剩约 60 G） |
| 驱动 / CUDA | 580.178.04 / CUDA 13.0（构建用 12.9，`/usr/local/cuda-12.9`） |
| 功耗上限 | GPU0 **145 W**、GPU1 **170 W**（`nvidia-power-limit.service` 开机应用；2026-11-16 核实，此前文档写 150/180 已过时）。**GPU1 165W 已实测否决**：同 boot A/B/A，decode 均值 71.5-72.0 vs 74.1-74.5 tok/s（**-2.6~-4%**）。**175W 也已测（交替 4+4 臂取中位）**：decode 中位 72.9 vs 73.5 tok/s（**-0.8%，噪声内无差异**；峰值仍摸到 174-177W）。**170W 即甜点位，两个方向都不必动**。功耗测试注意：连跑多臂有专家缓存热身漂移（后面的臂天然偏快），必须交替取中位，别信单臂 |
| ECC | `Enabled`，关掉实测省 0 MiB |
| 模型 | `/data/models/Qwen3.8-Flash-Next-UD-IQ4_XS-abliterated`（89 G），pack `/data/models/packs/ud-iq4_xs-abliterated`，MTP `/data/models/mtp/rt` |
| API | `http://<SERVER-IP>:8080/v1`，key `<API-KEY>`；网页版 `http://<SERVER-IP>:8080/` |

模型结构要点（影响调优判断）：48 层，每层 512 个专家，共 24,576 个 (层, 专家)；专家 gate/up 多数是 IQ3_XXS（28 层），down 多数是 Q5_0（33 层）、IQ4_NL（13 层）。稠密权重约 3.8 GiB（pack 1416 MiB + 原生矩阵 2462 MiB），HC（hyper-connection）权重 BF16 共 1212 MiB（源文件里是 Q8_0）。

---

## 2. 启动 / 停止 / 切换

**当前生产**：Strata 0.1.59.1 + 本仓库改动，脚本 `/data/models/strata-chenjx-q4`，配置 `~/.local/share/strata/install/strata-chenjx-q4.json`。

```sh
# 启动（必须先确认 8080 空闲）
cd /data/models
LOG=/data/models/logs/q4-$(date -u +%Y%m%d-%H%M%S).log
setsid nohup /data/models/strata-chenjx-q4 > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 48); do grep -q '^ready:' "$LOG" && break; sleep 10; done
tail -3 "$LOG"
```

**停止（安全写法，别用 pkill -f 直接匹配）**：

```sh
for pid in $(pgrep -f 'serve/server[.]py' 2>/dev/null); do
  exe=$(readlink /proc/$pid/exe 2>/dev/null || true)
  case "$exe" in
    *"/terminal/0.1.49.2/python/bin/python3.12"|*"/terminal/current/python/bin/python3.12") kill "$pid" ;;
  esac
done
for i in $(seq 1 30); do ss -lnt | grep -q ':8080 ' || break; sleep 1; done
```

> **坑**：`pkill -f 'serve/server.py'` 会匹配到发起它的 shell，把 SSH 会话一起杀掉。

**健康检查**：

```sh
curl -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer <API-KEY>'
nvidia-smi --query-gpu=index,memory.used,power.draw,power.limit --format=csv
```

**三个引擎（同一端口，切之前先停当前的）**：

| 脚本 | 说明 |
|---|---|
| `/data/models/strata-chenjx-q4` | 当前生产：0.1.59.1 + 本仓库改动 |
| `/data/models/start-strata-ud-iq4xs-official.sh` | 官方 0.1.59 组件引擎，配置 `strata-ud-iq4_xs-official.json`，测速对照用 |
| `/data/models/start-strata-ud-iq4xs-abliterated-v100fork.sh` | 旧 fork 0.1.45.7，仅作对照 |

### 当前配置（2026-11-13 04:56 UTC 启用 `STRATA_STAGE_TRIM=1` 并重启）

- `args`：`--layer-split 25`、`gpu=[1,0]`、`--max-context 262144`、`--kv int8`、`--kv-resident 32768`、`--pipeline-windows 2`、`--vram-reserve-mib/-later-mib 512/512`、`--spec 4 --spec-min-p 0.6`、`--mtp /data/models/mtp/rt`、`--suffix-draft 6`、`--pcie-frac 0.20`、`--pool-workers 14`、`--expert-cache auto`、`--expert-profile terminal/0.1.49.2/data/expert-profile.bin`、`--vision`。
- `env`：`STRATA_SM70_TABLE=0`、`STRATA_STAGE_TRIM=1`、`STRATA_PREFILL_CPU_SHARE=auto`、`STRATA_PREFILL_CPU_SHARE_MAX=6144`、`STRATA_PREFILL_STREAM_MIN=6144`、`STRATA_HC_Q8=0`、`STRATA_ROUTE_TAIL_SKIP=0`、`STRATA_AUX_CPUS=auto`、`STRATA_EMB_REUSE_ACCOUNT=1`、`STRATA_KV_GROW=1`、`STRATA_WATCHDOG_S=300`。
- `STRATA_PF_WMMA=1` / `STRATA_PF_WMMA2=1` **已启用**（2026-11-15 重评后入生产，见 §5.3；回退备份 `.bak-before-pfwmma2-20261115-132119`）。
- **质量否决（用户实测反馈，2026-11-15）**：`STRATA_SM70_TABLE=1` 实测智商下降，后续不再考虑整表或相关组合开关；保持 `STRATA_SM70_TABLE=0`，不得作为性能候选重试。
- 日志确认 sm_70 默认表开了 10 项（`SM70_TABLE` 被配置关掉）。
- trim 前备份：`strata-chenjx-q4.json.bak-before-stage-trim-20261113-045640`；更早备份：`.bak-before-pfwmma-20261112-130109`、`.bak-nsys`、`.bak-pcie`。**改配置前先备份。**
- **不要给生产或 pipelined verify 诊断配置加 `CUDA_LAUNCH_BLOCKING=1`**：引擎明确报告它与 verify window 不兼容（#1341），会让 prompt 卡到 watchdog，而不是帮助定位。

**这套配置就是“质量正常”的基准。** 后面每项提速都在它上面做 A/B。

---

## 3. 日志在哪

| 日志 | 路径 |
|---|---|
| wrapper（启动/ready/进度） | `/data/models/logs/<你起名>.log` |
| **引擎自身** | `~/.local/share/strata/install/strata-chenjx-q4.log`，**跨多次启动累积**，先找最后一行 `engine started` |
| 官方 | `~/.local/share/strata/install/strata-ud-iq4_xs-official.log` |

调优时最有用的几行：

```
strata generate: expert cache 4383 slots, 9.64 GiB of VRAM; policy is ... PROFILE, ranked by routing frequency, no eviction.
strata generate: layer split: CUDA1 runs layers 25-47, expert cache 3511 slots (8.36 GiB)
strata serve: prompt N tokens = R reused + M read in X ms (.. tok/s), G generated in Y ms (.. tok/s), drafts accepted A of B
strata serve: decode expert cache hit rate: 87.6% (...); ... (1.0% of all ... routed)   # 每请求，提速的核心指标
strata serve: CUDA1: 511 MiB of VRAM free with everything loaded - LOW ...              # 显存余量
```

---

## 4. 现状画像（2026-11-12 日志，最近 200 个请求，含不少测速请求）

| 指标 | 值 |
|---|---|
| 总耗时构成 | decode 约 6086 s，prefill 约 1128 s，**decode 占 84%** |
| decode 速度 | p10 64 / 中位 76 / p90 84 tok/s |
| 草稿接受率 | 中位 0.77（`--spec 4` + MTP + suffix draft） |
| decode 专家缓存命中率 | p10 76% / 中位 86% / p90 89%；没命中的几乎都由 CPU 算，PCIe 读只占约 1% |
| 缓存槽 | CUDA0 4383 + CUDA1 3511 = 7894 个，占全部专家的 32% |
| 缓存策略 | **（2026-11-15 更正，原"静态"描述有误）**启动按 profile 填满（日志 "PROFILE, no eviction" 只指初始填充）；**自适应换入层默认开启（阻塞式）**：`--adapt-every 4`、`--adapt-swaps 96`、decay 0.7、min-gain 1.5（generate.cpp:888/924/13187+，含换出围栏、交换缓冲、layer-split 感知）；`--adapt-async 1`（异步层，窗口间隙推进）未启用。服务器恢复后 grep 引擎日志 `adaptive tier` 看实际轮次/换入数再定调参方向 |
| 小块 prompt（< 2K token）读取 | p10 51 / 中位 130 / p90 413 tok/s，几十 token 的读取也要 1 秒多 |
| 显存 | CUDA1 加载完只剩 511 MiB（日志报 LOW），有一次请求开始时只剩 487 MiB |

之前的 profiling：decode 时 GPU 有 20% 时间在 `wait_flag_ge_kernel`（等 CPU 专家），CPU 专家池自旋 49%，多数没命中的专家一次只有 1 个 token。**decode 的瓶颈是没命中专家的 CPU 计算和逐层同步，提高命中率是最直接的提速手段。**

---

## 5. 下一步提速方向（按性价比排序）

### 5.1 已实测不采用：用真实负载重建专家 profile

- 现在用的是 terminal 0.1.49.2 自带的通用 profile，不是按这个 abliterated 模型和实际负载生成的；缓存又是静态的，排序直接决定命中率。命中率 p10 只有 76%，说明不同请求类型差别很大。
- 工具都现成：
  ```sh
  # 1) 收集路由轨迹（需停服务；每个提问一次引擎启动 + 约 10 s decode，45 个提问约 30 分钟）
  python tools/collect_profile_traces.py ~/.local/share/strata/install/strata-chenjx-q4.json ~/tune/traces \
         --corpus <自己的语料.json> --max-new 250 --mode routing
  # 2) 生成新排序（--reorder：轨迹决定前排，旧 profile 补齐剩余）
  python tools/make_profile_mix.py ~/tune/traces/c*.trace:1 --out ~/tune/expert-profile-workload.bin
  #    或 python tools/make_profile.py ~/tune/traces/*.trace --reorder --out ...
  # 3) 配置里把 --expert-profile 指向新文件（先备份配置），重启
  ```
- 语料要贴近实际使用（中文对话、agent/工具调用、代码），默认语料是 `bench/profile_corpus.json`。轨迹只记录生成部分的路由。
- 注意：`make_profile.py` 不加 `--reorder` 时，因为旧 profile 已经排满 24,576 个，轨迹不起作用（工具说明里有警告）。
- 数值影响：只改变哪些专家放在 GPU 上。GPU 与 CPU 算专家的舍入不同，回复可能有细微差别（日志写明“same quality”），这种差别现在本来就存在。
- 11-13 已收集 24 条中文 / Agent / 代码轨迹（22,793 个专家对），测试 mix、reorder、25% / 50% blend。mix 在短留出集上约 +1.9%，但完整矩阵三档 decode **下降 12.8–13.5%**；混合方案也在长上下文或短请求上退化。**这些候选均未进入生产，不要只凭短留出集重试该方向。**

### 5.2 已完成并入生产：层切分时去掉每张卡上重复的稠密权重

- 原配置每张卡都加载全部 48 层的稠密权重（约 3.8 GiB），其实只用自己那半层。`STRATA_STAGE_TRIM=1` 让 CUDA0 只加载 0–24、CUDA1 只加载 25–47 的稠密权重。
- 11-13 独立配置实测通过启动、graph capture、58-token 小请求、两轮 3.2K / 14K / 61K 矩阵、182 工具、4 道短题及 2,048-token 长输出；没有 CUDA error、Xid 或 repeated-reasoning 报警。随后仅给生产 env 增加 `STRATA_STAGE_TRIM=1` 并重启，健康检查通过。
- 常驻专家槽从 **7,894 增至 9,632（+22.0%）**。`expert-cache auto` 会把腾出的显存继续分给缓存，因此 LOW / 请求开始时的低显存告警仍可能出现，不代表 trim 失效。
- **不要在这个配置上设置 `CUDA_LAUNCH_BLOCKING=1`**：它会令 pipelined verify window 无法完成（引擎日志引用 #1341），300 秒 watchdog 后重启。初次探针的 58-token prompt 超时完全由该诊断变量造成；去掉后同一 trim 配置正常，无需源码修复。
- 数值影响：只改变权重驻留位置和缓存容量；缓存变大带来的 GPU/CPU 专家舍入差别同 5.1。实测数据见第 7 节。
- 11-13 已在 trim 上单独 A/B `STRATA_HC_Q8=1`：日志确认每 stage 使用 0.30–0.33 GiB 的原 GGUF Q8_0 HC，但 BF16 仍被其它路径保留，常驻专家从 **9,632 降为 9,294**。三轮 3.2K / 1,024-token 交替 decode **89.6 → 89.1（−0.56%，噪声内）**，无 CUDA/Xid；不采用，生产保持 `STRATA_HC_Q8=0`。

### 5.3 已采纳：PF_WMMA / PF_WMMA2（2026-11-15 重评入生产，prompt 14K/61K +4.2~4.5%）

- 11-12 实测：`STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1`（含今天加的 Q5_0 W2 tile）让 Q5_0 down 段 −46%，prompt 14K 1445 → 1515~1535、61K 2062 → 2134~2162 tok/s。
- 代价：这些专家层的 prompt 激活改为 FP16（超过 65504 会被截断），与默认不逐位一致。11-12 同时撤掉了它和 `SM70_TABLE`，没法确认当时的担心是否和它有关。
- 做法：单独打开它（不动 `SM70_TABLE`），跑第 6 节的回归集；没问题就保留。
- **11-15 已按此完成并采纳**：3 轮 × 3.4K/14K/61K off/on 交替（512 token、每臂重启、同一生产二进制仅 env 差异；`~/tune/results.jsonl` 标签 `pfwbase-r*` / `pfwon-r*`，进度 `~/tune/pfw-ab.progress`）。prompt 中位：3.4K 900.3→901.7（+0.2%，噪声内）、14K 1518.8→1583.1（**+4.2%**，配对 +3.9/+4.6/+4.2 三轮同向）、61K 2141.9→2238.7（**+4.5%**，配对 +4.4/+4.9/+4.3 三轮同向）；decode 3.4K −0.1% / 14K +4.1% / 61K +0.4%，无退化。
- 质量回归通过：182 工具返回结构化 `read_file {"path":"src/main.cpp"}` 与基线一致；4 短题 2 逐字相同 + 2 等价正确（同义改写/逻辑等价实现，属 GPU/CPU 专家舍入类既有差异）；2048 长输出无循环（`~/tune/q-pfwon*.json`）。
- 机制确认：config `env` 由 serve/server.py 注入**引擎子进程**（python 父进程 environ 看不到，验证时别查错进程）；引擎进程 environ 含两变量=1。无 error/Xid/stall，VRAM 与命中率与 off 臂同级（LOW 告警为既有状况）。
- 生产 11-15 13:22 UTC 重启生效，冒烟 14K 1562.5 / 61K 2238.2 tok/s，质量冒烟一致。回退 = 删 env 两行（备份见上）。`SM70_TABLE` 保持 0（质量否决不变）。

### 5.4 其他可试的旋钮（每个都要单独 A/B）

| 旋钮 | 现状 | 说明 |
|---|---|---|
| `--layer-split` | **25，不再调** | **用户指示（2026-11-15）：固定 25，不要再试 23/27 或其他值** |
| `--pool-workers` | 14 | 试 16，配合 `STRATA_AUX_CPUS` |
| `--pipeline-windows` | 2 | 试 0 对照 |
| `--adapt-async` | 0（阻塞式） | 异步自适应层：job 线程在 verify 窗口间隙推进换入轮（1Cat 同款结构）；arena 模式配 `STRATA_ADAPT_PICK_INLINE/ASYNC_WAIT/ASYNC_BATCH`；**非逐位一致（文档注明），必须过质量回归**；先看日志 `adapt wait` 占比再决定 |
| `--mtp-draft-vocab` | 默认 `rt/draft_vocab.bin` | 已有功能：draft 头只算子表行（验证仍全表）。服务器恢复后：`ls /data/models/mtp/rt/draft_vocab.bin` + 查日志子表大小；可自用语料重生成更贴负载的子表（A/B 接受率） |
| `--spec` | 4 | **6 已实测否决（2026-11-16）**：-2~-20%（q1 最差）；接受率降到 0.49/位（741/1512），第 5/6 草稿几乎不中被接受但每窗口多付验证计算。8 不必再试 |
| `STRATA_PF_GROUP_SCRATCH=1` + `STRATA_PF_HC_REGION=1` + `STRATA_TQ_SUB8=1` | 未设 | fork 实测 prompt −2~−6% 耗时，group scratch 改舍入 |
| `STRATA_SM70_TABLE=1` | **禁用** | **用户实测智商下降，质量否决，不再考虑或重试**；保持 0，相关组合开关也不作为后续候选 |
| `--vram-reserve-mib` / `-later-mib` | 512 / 512 | **不要再往下调**：日志已经建议 later 加到 769；5.2 做完后再看 |

### 5.5 更深的方向（需要较大改动）

**2026-11-15 已收逐窗口 GPU stage 分解**（`STRATA_VERIFY_PROFILE=1`+`STRATA_DECODE_TIMING=1` 临时注入重启采集，已还原；两 stage 各 14-16 ms GPU/窗口，wall 26-34 ms）。持续 decode（热缓存）下每窗口 GDN 层：稠密投影 q8+qkv gemv 1.46 + out-proj 0.60；专家命中路径 VRAM hits 2.10（最大单项真实计算）；miss 税 waitA 0.43+waitB 0.50+waitCPU 0.87+PCIe 0.20+copy 0.39 ≈ **2.4 ms 热 / 5.75 ms 冷**（冷窗口 19.5 ms 时 waitA 1.79+waitB 1.74+waitCPU 2.22）；**HC ≈ 2.2（GDN 层）+ 0.9（QSA 层）≈ 3.1 ms ≈ 窗口的 20%**；router+plan 0.63、shared+quant 0.75；QSA 全部 ≈ 3.0；stage1 另有 head 1.15。**HC 立项修正（2026-11-15 源码精读后）**：上文"1Cat 手法适用、砍半即 +5-10%"**说过头了，收回**。差距矩阵：①1Cat 的 HC 优化是 TP4 跨卡交换重叠（94 串行边界），本机 layer-split 整层单卡**无此结构**；②权重预取同思路（STRATA_GR_V3_PF）在 Blackwell 实测更慢（+17%），V3 读在 sm_86 实测 -3~-6%；③norm 已融合进 down；④staged 变体已有 cp.async 两 tile 流水。另 profile 的 hc0 norm 0.42 vs hc1 0.15 三倍差不是异常——hc0 的读还写 q8_1 图像（S26 QFUSE）。Strata HC 剩余空间是 GEMV 有效带宽（实测 ~35-40µs/半层 vs 带宽下限 ~15µs，约 2x 理论），属内核 tile 微调且邻近尝试为负。**零代码 A/B 已跑完（2026-11-16，每变体独立重启 + 同 5 题负载，基线 q0 74.3 / q1 80.5 / q2 69.3 / q3 85.4 / q5 74.6 tok/s）——全部否决，现配置即这三个旋钮的最优**：①`STRATA_SPEC_ADAPT_EVERY=2`+`WARM=200` 热身加速：**-3~-7% 一致负**（PCIe 双卡上早期换入代价>命中率收益，与 sm_86"未复现"历史一致），别再试；②`STRATA_GR_V3=1`：约 -1% 噪声内偏负（V100 与 sm_86 同为负，sm_120 专属成立），别再试；③`STRATA_HC_SPLIT=0`（plain）-3~-26%、`=1`（split）-3~-13%，**staged（默认/auto）最优保持**。**配置级红利确认归零；下一个方向只剩 VRAM hits 2.1 ms 的专家内核带宽（1Cat dp4a/QPN 系移植，真正的内核工程），或更深的 miss 路径重构。**

- **decode 单 token 专家的同步延迟**：CPU 池自旋 49%、GPU 等待 20%。要么减少 CPU 侧的逐层同步，要么让单 token 的缺失专家走 PCIe 由 GPU 算（`--pcie-frac` 扫过，0.20 已最优，单靠这个参数不行）。11-13 源码追踪确认 waitA / waitB / waitCPU 共用一个 spin kernel，旧 nsys 的 20% 不能直接归因于 CPU；上面 11-15 的分解就是这一步的产出，继续按它立项。
- **CUDA 门铃直写已实测不采用**：独立 worktree `~/strata-v100x2-doorbell` 放开 `STRATA_DOORBELL_STORE=1` 对 CUDA 的支持，复用已知序号直写内核，payload/序号 parity 通过。首轮非交替 decode 中位 +1.9%，但三轮 off/on 交替后只有 **90.1 → 90.3 tok/s（+0.22%，噪声内）**，无 CUDA/Xid/NEVER-RANG。未进生产；不要只凭首轮结果重试。
- **IQ4_XS gate/up 接入 EXPDEC 已实测不采用**：独立 worktree `~/strata-v100x2-expdec` 给 `Split<23>` 补 `apply_u`、无 grid 的 `EdGrid<23>`、白名单和模板分发；`iq_multi_parity` 真实 2560×640 几何及 `expdec_bench 23/20` 全逐位一致。该类型覆盖 3,350/9,632 个常驻专家，但 V100 双卡在 T=1/2 的 gate/up 均变慢，R2 更差；只有 T=4 real 约快 1–4%。生产平均 T≈2.7，因此没有做端到端，未进生产。Q5_0 down 不是 drop-in，未混入此补丁。
- **小块 prompt 的首 token 延迟**：58-token 与 90-token 请求走不同路径（默认 short_read=64；当前 verify S=6）。冷请求 783 ms 中，24 次 verify graph 懒 capture 合计约 219 ms，可在 READY 前做默认关闭的预热，但只改善首次请求。热路径优先测每层 doorbell、host/pool 和 drain；`STRATA_PREFILL_TIMING=1` 只覆盖 batched prefill，不包含 verify-window 外围的 loan/refill/checkpoint。

### 5.6 1Cat-vLLM 借鉴清单（2026-11-15 调研，仓库 D:\workspace\1Cat-vLLM，1CatAI/1Cat-vLLM@b7baf77，v1.5.3）

同模型（Qwen3.8-Flash-Next）同硬件（V100 SM70）的 vLLM 工程分支，下面按对本机的适配度排序。**跨卡带宽相关的借鉴全部要打折实测**：他们是 NV2 mesh，本机 PCIe（他们自己的 ring collectives 只认 NVLink 直连对，PCIe 回退 PyNCCL）。

**首查（可能破案 SM70_TABLE 质量否决，且修复几乎免费）**：1Cat 把 GDN 循环状态默认设为 **FP32**，源码注释原话（`vllm/model_executor/layers/mamba/mamba_utils.py:109-124`）："following the model dtype here can make MTP verification drift into self-consistent repetition after long outputs"（FP16 状态 → 长输出自洽重复退化）。本机 = Flash-Next（GDN 层）+ MTP + 长输出，与"实测智商下降"表现吻合。**第一步：查 Strata 的 GDN/SSM 状态 dtype；若跟随模型 FP16，切 FP32（状态块很小，256K 请求约 272.9 MiB 量级）。→ 2026-11-15 已查：`session.cpp:96` 的 `gdn_state_floats` 分配的就是 FP32 `float*`，该嫌疑排除，SM70_TABLE 质量下降另有原因，需走质量度量定位。** 两条配套 MTP 铁律（他们血泪换来）：draft 永远 PIECEWISE 图（捕获成 FULL 图会"collapse acceptance"，`llm_base_proposer.py:573`）；GDN 前向在 FULL 图下必须整体一个不透明边界（`VLLM_SM70_QWEN_GDN_FULL_FORWARD=1`），拆开编译即复现"接受率 1.000 + 重复流"。

**质量度量方法论**（替代主观"智商下降"判断）：digit_ratio（数字字符占比）/ weird_runs / max_digit_run / 输出 hash 对比 + **自然 EOS**（强制续写会污染失败信号）。完整实验矩阵见其根目录 `SM70_FLASH_V100_QUALITY_EXPERIMENT_LOG_20260615.md`（根因：CUDA graph 回放时元数据缓冲未刷新，**不是数值精度**——查 Strata 的开关时先排除同类 bug，对照 `gpu_input_batch.py:refresh_metadata` 模式）与 `SM70_MTP_OUTPUT_QUALITY_AUDIT_20260616.md`。

| 本机痛点 | 1Cat 方案 | 关键位置（相对 1Cat 仓库根） |
|---|---|---|
| 专家缓存命中率 p10 76%（**自适应换入层已存在**，2026-11-15 核实，见第 4 节更正；可借鉴的是启用 `--adapt-async`、调 decay/min-gain/every/swaps） | host ring + 每层 LRU 规划器 + 拷贝引擎 DMA 写回；"decode miss 由 host 线程拷入、step 在图内等待" | `csrc/gguf_expert_cache.cpp`、`gguf_shared_expert_{planner,dma}.*`、`vllm/config/kernel.py:546-610`（`expert_cache_fraction` 等） |
| miss 专家 CPU 池自旋 49% | 专家 80% GPU / 20% host 放置 + 小 miss 计划专用拷贝内核（提速 6-20%） | `docs/design/flashnext_decode_verifier_16gb_20261014.md` |
| 路由重复读专家（74% 专家单 token） | active-expert 紧凑分组 decode：重复 expert ID → offset 式多行 segment，已准入同模型族 | `docs/design/sm70_qwen38_awq_active_grouped_decode.md`，开关 `VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE` |
| MTP 接受率中位 0.77 | **已有同款（2026-11-15 核实）**：`--mtp-draft-vocab FILE`，默认 `<mtp>/draft_vocab.bin`（存在即用；data/draft_vocab_en.bin 40K 为例），draft 头只取子表行、验证仍全表（mtp.hpp:68-76, mtp.cpp:549 有接受率代价度量）。可做的是：确认生产子表生效与大小、用语料重生成更贴负载的子表 | `vllm/model_executor/models/qwen3_5_mtp_sm70_head.py`、`vllm/v1/spec_decode/{static_draft_vocab,draft_prob_alignment}.py`、`csrc/sm70_turbomind/ops/mtp_local_top1_sm70.cu` |
| 显存 LOW（曾剩 511 MiB） | 专家 bank 单设备分配（views+零尾）/ pinned host 池按 worker 合一（power-of-2 舍入浪费 81%）/ 13-bit 无损 draft 专家存储（FP16 正常值尾数 3 个零位，逐位不变，省 225 MiB/rank）/ 只捕获实际用的图、target/draft 捕获间释放 eager 块 | `docs/design/flashnext_16gb_host_kv.md`、`docs/design/sm70_memory_defaults.md` |
| KV 挤占专家显存 | host 权威历史 + GPU 热缓存（8-16K），NVLink 下实测代价仅 ~2.6%；**PCIe 代价未知需实测** | `docs/design/flashnext_16gb_host_kv.md`、`flashnext_host_fp8_kv.md` |
| HC 2.6-2.8 ms/stage（decode 第二大头） | down 权重前先发块 partials、交换期预取 up 权重、延迟全局 RMS、低秩 chunk 就绪发布；他们 HC 94 串行边界 ×32.9µs=3.09 ms/round，列为结构优化第一目标 | `docs/design/flashnext_persistent_execution.md`（HC 节）、`vllm/distributed/device_communicators/sm70_hc_ll.py` |
| 首 token 懒 capture 219 ms | 按上下文长度分桶预捕获 FULL 图（q=1 decode 按压缩+FP8-KV 分桶，MTP verify 显式桶覆盖最长请求） | `vllm/sm70_profiles/cudagraph_policy.py` |
| STRATA_* 开关已几十个 | Policy 字段系统：内核读解析好的 policy 值而非裸 env，engine-local 配置 | `csrc/sm70_policy.h` + `sm70_policy_fields.inc` |
| MMQ/wmma 分发靠手调 | GEMM 形状调优 LUT 可导入 | `csrc/sm70_turbomind/ops/gemm_runtime.cpp`，`VLLM_SM70_GEMM_LUT_PATH` |

**他们已否决/不适用，别从他们搬**：持久化 megakernel 只是提案无实测（MonoMoE/MPK/Hazy 都是 SM80+ 的 TMA/cp.async/WGMMA，V100 无此硬件）；FLUTE 式 LUT 复制（8 KB 表 × 32 份超 V100 共享内存）；TurboQuant KV 本机已排除（见第 7 节）；一切 TP4-NVLink 绝对数字。


---

## 6. 测量方法与质量回归

### 测速口径（否则数字不可比）

1. 每次测量前重启引擎，只测一次请求。
2. prompt 里加唯一 nonce，避开前缀缓存（返回里 `cached` 为 0）。
3. `temperature=0`、固定 `max_tokens`（decode 至少 256，建议 512–1024），看 `timings.prompt_ms` / `predicted_per_second`，同时记日志里的命中率。
4. 至少 3.2K / 14K / 61K 三档，各 2–3 次取中位。
5. 脚本：`~/tune/bench.py`。

### 质量回归（每项改动都跑）

- **定量度量（2026-11-15 新增，1Cat 方法论）**：`python tools/output_quality.py 回复文件... --json run.json` 打 digit_ratio / max_digit_run / num_repeat_runs / weird_runs / weird_share / sha256（退出码 0/1/2），`--compare base.json cand.json` 对比两轮配置。阈值已按本机健康产出校准（纯标点重复/整行代码重复/短回复数字占比不算退化；HTML 的 ~2.6% 数字不算 WARN）。**基线已收**：`D:\workspace\strata-quality\baseline.json`（5 题：数学/KV cache/代码/中文短文/长 HTML，greedy），采集脚本 `D:\workspace\strata-quality\collect.py`。改动后同题重收，`--compare` 看漂移。
- 182 工具的 agent 请求：必须返回结构化 `tool_calls`（之前是 `read_file(src/main.cpp)`）。
- 4 道短题：答案与基准一致（措辞可以不同）。
- 1 道长输出题：看是否出现循环、生成 token 数是否异常。
- 注意 `STRATA_PREFILL_CPU_SHARE=auto` 按计时分配，同一请求两次输出可能略有不同；需要逐字比较时临时设 `STRATA_PREFILL_CPU_SHARE=0`。
- 评估质量时请求里显式传采样参数：客户端不传 temperature 时 server 默认贪心（`serve/server.py:955-958`），Qwen 官方建议思考模式 temperature 0.6 / top_p 0.95 / top_k 20。

---

## 7. 已做过什么（别重复踩）

### 源码提交（`~/strata-v100x2-git`，main，领先 origin 8 个，**都没推送**）

| commit | 内容 | 数值影响 |
|---|---|---|
| `9ee32a71` | sm_70 默认开关表，11 项；`STRATA_SM70_PERF=0` 整表关 | 10 项注释写逐位一致；`SM70_TABLE` 原作者写“要在 V100 上确认后再默认开”，生产已用配置关掉 |
| `e8912f0d` | 修启动日志标签 | 无 |
| `e0b5ca5e` | QSA prompt top-k 在 Volta 用 active-block bound | 代码推理同 id；`STRATA_TOPK_CAPACITY_GUARD=1` 退回 |
| `650c180d` | Q5_0 MMQ 默认编进（CMake `STRATA_ORCA_Q4KS_MMQ` 默认 ON）；W2 WMMA tile 支持 Q5_0；AVX2 多 token Q5_0 down；parity 测试 | Q5_0 MMQ 激活 q8_1（与 llama.cpp 一致），只能编译期关；W2 opt-in；AVX2 与 ggml 逐位相同，`STRATA_NO_Q50_256=1` 关 |
| `65563a55` / `5e66e9ad` / `1164d233` / `ebb1642b` | 文档 | 无 |

推送：GitHub 403（PAT 只读）。另外服务器上**没有配置 git 身份**，`git commit` 会报 “Author identity unknown”；本次文档改动已 `git add` 但没提交，需要用户用自己的身份提交。

### 实测数据

11-12 上午（Q5_0 MMQ 之前，2K/8K/32K）：

| 配置 | prefill tok/s | decode tok/s |
|---|---|---|
| 官方 0.1.59 | 812 / 1423 / 2044 | 44.4 / 54.5 / 57.7 |
| 表开，档 B 关 | 821 / 1444 / 2066 | 42.1 / 55.3 / 62.2 |
| + `PF_WMMA=1` | 836 / 1466 / 2097 | 40.7 / 63.1 / 80.4 |
| + `SPLIT_OWN=1` | **734 / 1066 / 1154** | 37.8 / 52.6 / 60.0 |
| 表关 `SM70_PERF=0` | 820 / 1426 / 2045 | 39.1 / 53.8 / 62.5 |

11-12 下午（`~/tune/bench.py`，512 token 输出）：

| 配置 | prompt 14K | prompt 61K |
|---|---|---|
| Q5_0 MMQ 之前 | 1364 / 1388 / 1381 / 1279 | 1960 / 1959 |
| Q5_0 MMQ（现默认） | 1424 / 1483 / 1489 / 1473 | 2094 / 2086 / 2098 |
| + `PF_WMMA=1 PF_WMMA2=1`（含 Q5_0 W2） | 1515 | 2134 / 2078 |
| 同上，生产开启时 | 1535 | 2162 |

decode（3.2K/14K，1024 token 输出）约 78–80 tok/s。细节见 `docs/V100_PORT_CANDIDATES.md`。

**结论**：sm_70 表约 +1%；Q5_0 MMQ prompt +5~10%；Q5_0 进 W2 再 +3.5~5%；`--pcie-frac` 0.20 最优（0.00–0.20 持平，0.35 起变慢，auto 收敛 0.15）；AVX2 Q5_0 对 decode 无可测收益；`SPLIT_OWN` 在 16G 上 prompt −26~−44%。

11-13 `STRATA_STAGE_TRIM=1`，生产二进制与原 profile，每档独立重启两次取中位：

| prompt 档 | prefill 原 → trim | 变化 | decode 原 → trim | 变化 | 命中率原 → trim |
|---|---:|---:|---:|---:|---:|
| 3.2K | 825.9 → 890.0 | +7.77% | 77.2 → 86.7 | +12.31% | 85.6% → 88.7% |
| 14K | 1482.8 → 1525.3 | +2.87% | 75.3 → 82.8 | +9.89% | 86.7% → 90.8% |
| 61K | 2120.2 → 2172.4 | +2.46% | 68.9 → 71.2 | +3.34% | 87.8% → 90.7% |

原始结果在 `~/tune/results.jsonl`（`lossless-base-*` 与 `stage-trim-*`），trim 日志在 `~/tune/stage-trim-noblocking.engine.log`，长输出在 `~/tune/q-stage-trim-long.json`。质量回归通过，生产已启用。trim 落地本身没有修改 C++/CUDA 源码，也没有提交或推送。

11-13 门铃直写源码探测：`verify_parity` 新增 doorbell payload/序号对照并 PASS；off/on 原始结果同样在 `~/tune/results.jsonl`（`doorbell-off-*`、`doorbell-on-*`、`doorbell-ix-*`）。交替三轮 decode +0.22%，不采用。实验补丁保留在独立 worktree，未提交、未推送，生产仍用原 `build-v100/strata`。

11-13 原生产二进制加诊断的分阶段数据在 `~/tune/decode-phase-profile.engine.log`：28–30 ms/window 中 GPU-reach wait 约 14.6–15.5 ms、per-layer host 约 7.0–7.4 ms；waitCPU 约 0.9–1.4 ms/stage，HC 约 2.6–2.8 ms/stage。wait 含跨 stage 服务，不能相加成 wall。

11-13 Q8 HC 结果标签为 `hcq8-ix-*`，日志 `~/tune/hcq8-{off,on}.engine.log`；交替 decode −0.56%，不采用。IQ4_XS EXPDEC parity / bench 分别保存为 `~/tune/expdec-iq4xs-parity.txt`、`~/tune/expdec-iq4xs-bench.txt`；全逐位一致但相关 T 变慢，不采用。所有实验均未提交或推送。

11-15 `STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1` 重评（3 轮交替，详见 §5.3）：prompt 14K +4.2% / 61K +4.5%（各三轮同向）、3.4K 持平；decode 无退化；质量回归通过，**已入生产**（env 备份 `.bak-before-pfwmma2-20261115-132119`）。实验 config 副本 `~/tune/pfw-off.json` / `pfw-on.json`（含各自 engine log）。

### 已排除（别再测）

- all-cards split engine：模型的 Q5_0 down 不被支持。
- TurboQuant KV、ReVolta 分段 QSA、`--batch-mtp`/`--batch` 跑 262144、L2/L3 会话缓存、关 ECC。
- `--numa-hot-node`、`POOL_SPIN_ZERO`、`GDN_REC_FAST`、`EXPERT_PAIR`、`VERIFY_ECAP`、`INPUT_ANCHOR_DEFER`：0.1.59.1 里不存在。
- PLE 锁内存、`STRATA_MTP_STAGE0=1`（decode −18%）、`STRATA_PIPELINE_THETA`、`PREFILL_AUTO_MAX=16384`+`SPLIT_CHUNK16K`、`PREFILL_HELP`/`PREFILL_PIPE`、`STRATA_SPLIT_MTP_BATCH=1`（decode −4~6%）。
- `--kv-resident` 下调：32768 cell 在 VRAM 只占约 0.2 GiB，最多多 1% 槽。
- `STRATA_GDN_CHUNK`（GDN 分块 wmma）：状态存成 FP16，parity 容差 rel 50，**不要开**；`STRATA_GDN_REC_WARP=2` 改求和顺序，收益 +6~9%（只在 prompt 的 GDN 段），没在本机测过，可列为低优先级候选。

---

## 8. 仓库、构建与其它坑

- 源码：`~/strata-v100x2-git`（0.1.59.1 + 上表改动），生产构建 `build-v100`（`build-v100/strata`）。
- 构建：`cmake -S . -B build-v100 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`，增量 `cmake --build build-v100 -j18`（全量 15–20 分钟）。**试验性改动另建构建目录**，别动生产的 `build-v100`。
- 测试构建：另建目录加 `-DSTRATA_BUILD_TESTS=ON`；`expdec_bench` 等是 `EXCLUDE_FROM_ALL`，要 `--target` 单独编。GPU 测试需要停服务（显存被占满）。
- 官方引擎：`~/.local/share/strata/components/engine/0.1.59-linux-x86_64-cuda-cuda12/strata`（另有 0.1.55、0.1.57），也是 `STRATA_EXPERIMENTAL_SM60=ON` 构建。
- 文档：`docs/V100_PORT_CANDIDATES.md`（移植清单 + 实测）、`docs/HANDOFF-V100-TUNING.md`（本文）；本地副本 `D:\workspace\V100-TUNING-HANDOFF.md`。
- 服务器时间 UTC；`date -u` 打日志名。
- `github.com` 直连时通时不通；全局 `url.insteadOf` 改写到 `gh-proxy.com`，push 会被凭据域不匹配挡住。
- 8080 只能跑一个实例。
- 启动脚本注释说档 A“全部 bitwise/parity”，对 `SM70_TABLE` 不成立。
- `STRATA_PREFILL_PROMOTE`（官方配置里有）在任何二进制里都不存在，无效。

## 9. 2026-11-13 晚间补充：basalt 调研衍生的四项实测（全部未采用）

对照 basalt（Strata 家族 fork）与 GitHub 调研做的 A/B，全部 3 轮 × 3.4K/14K/61K 交替、512 token、
每臂重启（数据 `~/tune/results.jsonl`，标签 min1-/wide-/dw4-/dw2-/base-r*）：

| 实验 | 3.4K / 14K / 61K decode（vs 同轮 base 中位） | 结论 |
|---|---|---|
| `STRATA_PCIE_MIN1=1`（miss≥2 保底 1 个走 PCIe） | −2.7% / −8.4% / −9.7% | **不采用**。命中率 90→93%、PCIe 0.4→2% 机制生效，但 GPU 经 PCIe 算 miss 比 CPU 贵（PHB 拓扑 + mode-2 copy kernel）。与老扫参 pcie-frac 0.35 变慢一致 |
| `STRATA_GR_UP_WIDE=1`（up 读一 warp 一行，160→1280 block，上游已写好、逐位） | +1.4% / −2.0% / +1.3% | **不采用**，噪声内无信号 |
| `STRATA_GR_DOWN_WARPS=4`（down 读 41→81 block，逐位，worktree dw-grid） | −1.3% / −4.2% / +1.3% | **不采用**；prefill 还略亏（block 翻倍 = 激活 staging 冗余翻倍） |
| `STRATA_GR_DOWN_WARPS=2`（162 block） | −6.3% / −5.0% / −7.6% | **不采用** |

**NEXT-OPT-HC-READ 的 A 路线至此三种形态全部实测否决**（HC_PERSIST 有静默写坏风险；up_wide 无信号；
down grid 负收益）：T≈3 的 HC 读不是网格饥饿，改占用率在此机不兑现。HC 这 2.71 ms 若再动，
只剩"每窗口 CPU 侧 gap"一类未验证方向。

其它核对结论（改写本文档部分旧表述）：
- 专家 arena **早已全量 pin**（57.6 GiB 一次 cudaHostRegister PORTABLE|Mapped 成功），不存在"提高 pin 上限"的空间。
- **自适应换层默认就是开的**（`--adapt-every 4`/`--adapt-swaps 96` 默认值，generate.cpp:20580 分配 usage；
  §4 "缓存策略是静态"的表述不对——日志 "no eviction" 只描述 ExpertCache 本身）。
- `STRATA_ADAPT_FETCH` 与 `STRATA_PCIE_BALANCE` 被 `!multi_gpu` 硬禁（generate.cpp:20582/4849），
  层切分下无法启用。
- PCIe 0.6% 的根因：`m=(nmiss×51)>>8`，nmiss≤5 恒 0——设计行为，不是 bug。

实验资产：worktree `~/strata-v100x2-dw`（分支 dw-grid，含 STRATA_GR_DOWN_WARPS 补丁，
gr_parity 6 组合全 PASS、fused_gr_bench 0 differing；**未提交**）；生产二进制从未被长期替换
（A/B 后已恢复原文件）。不要重复测以上四项。

### 9.1 采纳：STRATA_FS_SLOTS=2（2026-11-14 入生产）

- **来源**：ForesightSwap 休眠件（#1348），由 basalt/AdapMoE 的 just-missed 重用思路激活实测。
- **数据**（3 轮 × 3 档交替，同二进制仅 env 差异）：3.4K +1.2% / 14K −1.4% / **61K +4.7%（76.3 vs 72.9 中位，三轮 3/3 同向）**。收益随上下文变长而增大——正是 miss 集中的地方；每请求 4400~6500 个专家从 swap 空间按 GPU 命中服务。
- **质量**：182 工具结构化 tool_calls 与基线一致；4 短题 3 逐字相同 1 同义改写（GPU/CPU 专家舍入类既有差异）；2048 长输出无循环。
- **代价**：每卡 ~112 MiB 自有显存（CUDA1 余 399 MiB，LOW 告警原本就有，未触发 stall 阈值 256）。
- **FS_SLOTS=4 已分诊否决**（61K 71.4 vs FS2 77.3：槽翻倍挤掉专家缓存，不要再加）。
- 配置备份 `strata-chenjx-q4.json.bak-before-fs-*`；关掉=删 env 里 STRATA_FS_SLOTS。
- router look-ahead 预测（STRATA_FS_AHEAD）本轮贡献为 0（"0 from the router look-ahead"）——全部收益来自 just-missed 时间重用；预测式预取未证伪，留作后续。
