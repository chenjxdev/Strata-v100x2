# 交接文档：Strata V100 提速调优（给下一个 agent）

最后更新：2026-11-13 (Asia/Shanghai)。

**目标：在尽量不降低生成质量的前提下提速。** 用户确认当前生产配置的质量正常，不需要排查质量问题。每项提速改动做完后，用第 6 节的回归集确认质量没有变差即可。

---

## 1. 机器与现状

| 项 | 值 |
|---|---|
| 主机 | `chenjx-server`，Ubuntu 22.04，**服务器时钟是 UTC**（比北京慢 8 小时） |
| SSH | `chenjx@172.16.3.88:22`，密码 `5419`（sudo 同密码） |
| GPU | 2 × Tesla V100-SXM2-16GB（sm_70，各 Gen3 x16） |
| GPU 互连 | **NVLink 全部 inactive**，拓扑 `PHB`（走 CPU 的 PCIe 桥），`cudaDeviceCanAccessPeer` 双向可用 |
| CPU / 内存 | Xeon E5-2686 v4，18 核 ×2 SMT = 36 线程，单 NUMA，125 GB RAM，**有 AVX2，没有 AVX-512 / VNNI** |
| 磁盘 | 单块 256 G NVMe（`/`，剩约 60 G） |
| 驱动 / CUDA | 580.178.04 / CUDA 13.0（构建用 12.9，`/usr/local/cuda-12.9`） |
| 功耗上限 | GPU0 **150 W**、GPU1 **180 W**（`nvidia-power-limit.service` 开机应用） |
| ECC | `Enabled`，关掉实测省 0 MiB |
| 模型 | `/data/models/Qwen3.8-Flash-Next-UD-IQ4_XS-abliterated`（89 G），pack `/data/models/packs/ud-iq4_xs-abliterated`，MTP `/data/models/mtp/rt` |
| API | `http://172.16.3.88:8080/v1`，key `chenjx-v100`；网页版 `http://172.16.3.88:8080/` |

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
curl -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer chenjx-v100'
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
- `STRATA_PF_WMMA` / `PF_WMMA2` 目前**没开**（11-12 开过，后来撤掉）。
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
| 缓存策略 | **静态**：启动时按 profile 填满，不换出，没有自适应换入 |
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

### 5.3 重新评估 PF_WMMA / PF_WMMA2（prompt +3.5~5%）

- 11-12 实测：`STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1`（含今天加的 Q5_0 W2 tile）让 Q5_0 down 段 −46%，prompt 14K 1445 → 1515~1535、61K 2062 → 2134~2162 tok/s。
- 代价：这些专家层的 prompt 激活改为 FP16（超过 65504 会被截断），与默认不逐位一致。11-12 同时撤掉了它和 `SM70_TABLE`，没法确认当时的担心是否和它有关。
- 做法：单独打开它（不动 `SM70_TABLE`），跑第 6 节的回归集；没问题就保留。

### 5.4 其他可试的旋钮（每个都要单独 A/B）

| 旋钮 | 现状 | 说明 |
|---|---|---|
| `--layer-split` | 25 | 试 23 / 27：GPU0 被 150 W 压着；配合 5.2 后再看 |
| `--pool-workers` | 14 | 试 16，配合 `STRATA_AUX_CPUS` |
| `--pipeline-windows` | 2 | 试 0 对照 |
| `--spec` | 4 | 试 6/8；只影响草稿窗口，输出经验证不变，但可能更慢 |
| `STRATA_PF_GROUP_SCRATCH=1` + `STRATA_PF_HC_REGION=1` + `STRATA_TQ_SUB8=1` | 未设 | fork 实测 prompt −2~−6% 耗时，group scratch 改舍入 |
| `STRATA_SM70_TABLE=1` | 0 | 专家层 mode 8 等；表整体只测到约 +1%，mode 8 在 V100 上没有逐位验证记录，优先级低 |
| `--vram-reserve-mib` / `-later-mib` | 512 / 512 | **不要再往下调**：日志已经建议 later 加到 769；5.2 做完后再看 |

### 5.5 更深的方向（需要较大改动）

- **decode 单 token 专家的同步延迟**：CPU 池自旋 49%、GPU 等待 20%。要么减少 CPU 侧的逐层同步，要么让单 token 的缺失专家走 PCIe 由 GPU 算（`--pcie-frac` 扫过，0.20 已最优，单靠这个参数不行）。11-13 源码追踪确认 waitA / waitB / waitCPU 共用一个 spin kernel，旧 nsys 的 20% 不能直接归因于 CPU；下一步先用现有 `STRATA_DECODE_TIMING=1` 和 `STRATA_VERIFY_PROFILE=1` 分解，不再盲调。
- **CUDA 门铃直写已实测不采用**：独立 worktree `~/strata-v100x2-doorbell` 放开 `STRATA_DOORBELL_STORE=1` 对 CUDA 的支持，复用已知序号直写内核，payload/序号 parity 通过。首轮非交替 decode 中位 +1.9%，但三轮 off/on 交替后只有 **90.1 → 90.3 tok/s（+0.22%，噪声内）**，无 CUDA/Xid/NEVER-RANG。未进生产；不要只凭首轮结果重试。
- **IQ4_XS gate/up 接入 EXPDEC 已实测不采用**：独立 worktree `~/strata-v100x2-expdec` 给 `Split<23>` 补 `apply_u`、无 grid 的 `EdGrid<23>`、白名单和模板分发；`iq_multi_parity` 真实 2560×640 几何及 `expdec_bench 23/20` 全逐位一致。该类型覆盖 3,350/9,632 个常驻专家，但 V100 双卡在 T=1/2 的 gate/up 均变慢，R2 更差；只有 T=4 real 约快 1–4%。生产平均 T≈2.7，因此没有做端到端，未进生产。Q5_0 down 不是 drop-in，未混入此补丁。
- **小块 prompt 的首 token 延迟**：58-token 与 90-token 请求走不同路径（默认 short_read=64；当前 verify S=6）。冷请求 783 ms 中，24 次 verify graph 懒 capture 合计约 219 ms，可在 READY 前做默认关闭的预热，但只改善首次请求。热路径优先测每层 doorbell、host/pool 和 drain；`STRATA_PREFILL_TIMING=1` 只覆盖 batched prefill，不包含 verify-window 外围的 loan/refill/checkpoint。

---

## 6. 测量方法与质量回归

### 测速口径（否则数字不可比）

1. 每次测量前重启引擎，只测一次请求。
2. prompt 里加唯一 nonce，避开前缀缓存（返回里 `cached` 为 0）。
3. `temperature=0`、固定 `max_tokens`（decode 至少 256，建议 512–1024），看 `timings.prompt_ms` / `predicted_per_second`，同时记日志里的命中率。
4. 至少 3.2K / 14K / 61K 三档，各 2–3 次取中位。
5. 脚本：`~/tune/bench.py`。

### 质量回归（每项改动都跑）

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

### 已排除（别再测）

- all-cards split engine：模型的 Q5_0 down 不被支持。
- TurboQuant KV、ReVolta 分段 QSA、`--batch-mtp`/`--batch` 跑 262144、L2/L3 会话缓存、关 ECC。
- `--numa-hot-node`、`POOL_SPIN_ZERO`、`GDN_REC_FAST`、`EXPERT_PAIR`、`VERIFY_ECAP`、`INPUT_ANCHOR_DEFER`：0.1.59.1 里不存在。
- PLE 锁内存、`STRATA_MTP_STAGE0=1`（decode −18%）、`STRATA_PIPELINE_THETA`、`PREFILL_AUTO_MAX=16384`+`SPLIT_CHUNK16K`、`PREFILL_HELP`/`PREFILL_PIPE`、`STRATA_SPLIT_MTP_BATCH=1`（decode −4~6%）。
- `--kv-resident` 下调：32768 cell 在 VRAM 只占约 0.2 GiB，最多多 1% 槽。
- `STRATA_GDN_CHUNK`（GDN 分块 wmma）：状态存成 FP16，parity 容差 rel 50，**不要开**；`STRATA_GDN_REC_WARP=2` 改求和顺序，收益 +6~9%（只在 prompt 的 GDN 段），没在本机测过，可列为低优先级候选。

---

## 8. 仓库、构建与其它坑

- 源码：`/home/chenjx/strata-v100x2-git`（0.1.59.1 + 上表改动），生产构建 `build-v100`（`build-v100/strata`）。
- 构建：`cmake -S . -B build-v100 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`，增量 `cmake --build build-v100 -j18`（全量 15–20 分钟）。**试验性改动另建构建目录**，别动生产的 `build-v100`。
- 测试构建：另建目录加 `-DSTRATA_BUILD_TESTS=ON`；`expdec_bench` 等是 `EXCLUDE_FROM_ALL`，要 `--target` 单独编。GPU 测试需要停服务（显存被占满）。
- 官方引擎：`~/.local/share/strata/components/engine/0.1.59-linux-x86_64-cuda-cuda12/strata`（另有 0.1.55、0.1.57），也是 `STRATA_EXPERIMENTAL_SM60=ON` 构建。
- 文档：`docs/V100_PORT_CANDIDATES.md`（移植清单 + 实测）、`docs/HANDOFF-V100-TUNING.md`（本文）；本地副本 `D:\workspace\V100-TUNING-HANDOFF.md`。
- 服务器时间 UTC；`date -u` 打日志名。
- `github.com` 直连时通时不通；全局 `url.insteadOf` 改写到 `gh-proxy.com`，push 会被凭据域不匹配挡住。
- 8080 只能跑一个实例。
- 启动脚本注释说档 A“全部 bitwise/parity”，对 `SM70_TABLE` 不成立。
- `STRATA_PREFILL_PROMOTE`（官方配置里有）在任何二进制里都不存在，无效。
