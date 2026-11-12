# 交接文档：Strata V100 调优（给下一个 agent）

最后更新：2026-11-12 18:00 (Asia/Shanghai)，会话交接。

---

## 1. 机器与现状

| 项 | 值 |
|---|---|
| 主机 | `chenjx-server`，Ubuntu 22.04，**服务器时钟是 UTC**（比北京慢 8 小时，日志时间戳按 UTC 读） |
| SSH | `chenjx@172.16.3.88:22`，密码 `5419`（sudo 同密码） |
| GPU | 2 × Tesla V100-SXM2-16GB（sm_70，各 Gen3 x16） |
| GPU 互连 | **NVLink 全部 inactive**，拓扑 `PHB`（走 CPU 的 PCIe 桥），`cudaDeviceCanAccessPeer` 双向可用 |
| CPU / 内存 | Xeon E5-2686 v4，单路 18 核 ×2 SMT = 36 线程，单 NUMA 节点，125 GB RAM |
| 磁盘 | 单块 256 G NVMe（`/`，剩约 60 G） |
| 驱动 / CUDA | 580.178.04 / CUDA 13.0（构建用 12.9，`/usr/local/cuda-12.9`） |
| 功耗上限 | GPU0 **150 W**、GPU1 **180 W**，由 `nvidia-power-limit.service` 开机应用（脚本 `/usr/local/sbin/nvidia-apply-power-limits.sh`，带读回校验/重试，无定时器） |
| ECC | 当前 `Enabled`，`pending=Disabled`；**实测关掉省 0 MiB**（total 16384→16384），仅为满足要求留了 pending |
| 模型 | `/data/models/Qwen3.8-Flash-Next-UD-IQ4_XS-abliterated`（89 G），pack `/data/models/packs/ud-iq4_xs-abliterated`，MTP `/data/models/mtp/rt` |
| API | `http://172.16.3.88:8080/v1`，key `chenjx-v100`；网页版 `http://172.16.3.88:8080/` |

---

## 2. 启动 / 停止 / 切换

**当前默认引擎（推荐）**：Strata 0.1.59.1 + sm_70 默认表，脚本 `/data/models/strata-chenjx-q4`。

```sh
# 启动（必须先确认 8080 空闲）
cd /data/models
LOG=/data/models/logs/q4-$(date -u +%Y%m%d-%H%M%S).log
setsid nohup /data/models/strata-chenjx-q4 > "$LOG" 2>&1 < /dev/null &
# 等待就绪（约 70–120 s）
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

> **坑**：`pkill -f 'serve/server.py'` 会匹配到发起它的 shell（命令行里含同样字符串），把 SSH 会话一起杀掉。必须按 `/proc/<pid>/exe` 过滤，或把模式写成 `server[.]py`。

**健康检查**：

```sh
curl -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer chenjx-v100'
nvidia-smi --query-gpu=index,memory.used,power.draw,power.limit --format=csv
```

**其它两个引擎（同一端口，切之前先停当前的）**：

| 脚本 | 说明 |
|---|---|
| `/data/models/strata-chenjx-q4` | 当前主力：0.1.59.1 + sm_70 表，配置 `strata-chenjx-q4.json` |
| `/data/models/start-strata-ud-iq4xs-official.sh` | 官方 0.1.59 组件引擎，配置 `strata-ud-iq4_xs-official.json` |
| `/data/models/start-strata-ud-iq4xs-abliterated-v100fork.sh` | 旧 fork 0.1.45.7（**prefill 偏移 bug 已修**，仅作对照，勿用于生产） |

**当前配置**（`~/.local/share/strata/install/strata-chenjx-q4.json`）：
`exe=~/strata-v100x2-git/build-v100/strata`、`layer_split="25"`、`gpu=[1,0]`、`--max-context 262144`、`--kv int8`、`--kv-resident 32768`、`--pipeline-windows 2`、`--vram-reserve-mib/-later-mib 512/512`、`--spec 4`、`--pcie-frac 0.20`、`--pool-workers 14`。

---

## 3. 日志在哪

| 日志 | 路径 |
|---|---|
| wrapper（启动/ready/进度） | `/data/models/logs/<你起名>.log` |
| **引擎自身**（路径选择、缓存槽、KV、错误） | `~/.local/share/strata/install/strata-chenjx-q4.log` —— **注意它跨多次启动累积**，`tail` 前先看时间戳 |
| 旧 fork | `~/.local/share/strata/install/strata-ud-iq4_xs-abliterated-v100fork.log` |
| 官方 | `~/.local/share/strata/install/strata-ud-iq4_xs-official.log` |

引擎日志里最有用的几行：

```
strata: sm_70 (Volta): 11 exact speed switches on by default (...)      # 表是否生效
strata generate: layer split across GPUs [1, 0] (25)                    # 实际切分
strata generate: KV streaming: 32768 of 262144 cells per QSA layer ...
strata generate: expert cache auto: 10.67 GiB free, 512 MiB reserved -> 3612 slots
strata serve: KV streaming: ... hit VRAM, ... read from RAM             # 每请求后
```

---

## 4. 已做过什么、结论是什么（别重复踩）

### 已提交到源码（仓库 `~/strata-v100x2-git`，分支 main，领先 origin 6 个提交）

| commit | 内容 |
|---|---|
| `9ee32a71` | **sm_70 默认开关表**：11 项，逐项有本仓库内 bitwise/parity 依据；`STRATA_SM70_PERF=0` 整表关；受 `STRATA_DEFAULTS=026` 抑制 |
| `e8912f0d` | 修启动日志（原先把 Volta 标成 sm_60、给错的总开关名） |
| `e0b5ca5e` | **QSA top-k 在 Volta 用 active-block bound**（移植 jmnargi PR#52，同 ids） |
| `5e66e9ad` / `65563a55` / `1164d233` | 实测数据、候选清单、split 引擎结论（文档） |

推送状况：**push 被 GitHub 403 拒绝**（PAT 只有读权限）；本地有 `/home/chenjx/v100-changes.patch`（`git am` 可用）和 `.bundle`。

### 实测基线（2×16G，IQ4_XS，split 25，262144，T=0，唯一 nonce 避开前缀缓存）

| 配置 | prefill tok/s (2K/8K/32K) | decode tok/s (2K/8K/32K) |
|---|---|---|
| 官方 0.1.59 | 812 / 1423 / 2044 | 44.4 / 54.5 / 57.7 |
| **当前（表开，档 B 关）** | **821 / 1444 / 2066** | 42.1 / 55.3 / 62.2 |
| + `PF_WMMA=1` | 836 / 1466 / 2097 | 40.7 / 63.1 / 80.4 |
| + `SPLIT_OWN=1` | **734 / 1066 / 1154** | 37.8 / 52.6 / 60.0 |
| 表关 `SM70_PERF=0` | 820 / 1426 / 2045 | 39.1 / 53.8 / 62.5 |
| QSA bound 旧规则（`TOPK_CAPACITY_GUARD=1`） | 837 / 1442 / 2061 | 40.4 / 60.4 / 78.3 |

**结论**：sm_70 表 ≈ **+1%**（fork 的 +11.4% 是 32G 卡 + IQ3_S + split 16）；`SPLIT_OWN` 在 16G 上 **−26~−44% prefill**（抢 1.9 GiB/tage 导致缓存 3612→3056 槽）；`PF_WMMA` 对 IQ4_XS 基本无效；QSA bound 无差异。

### 已排除

- **all-cards split engine**（paskaa 的 +57% 路线）：sm_70 能编（`build-split/src/split/strata-split-2`），但**模型不支持**：`blk.0.ffn_down_exps.weight is ggml type 6 ... use the normal layer split`。要换 sfnx 支持的权重才有戏。
- TurboQuant KV（prefill −20%）、ReVolta 分段 QSA（满上下文 decode −7~16%）、`--batch-mtp`/`--batch` 跑 262144（专家缓存 4758→658 槽）、L2/L3 会话缓存那套（0.1.59.1 架构不同）、`STRATA_STAGE_TRIM`（Xid 31 必崩）。
- **关 ECC**：实测省 0 MiB，只丢纠错。
- `--numa-hot-node`、`POOL_SPIN_ZERO`、`GDN_REC_FAST`、`EXPERT_PAIR`、`VERIFY_ECAP`、`INPUT_ANCHOR_DEFER` 等在 0.1.59.1 里引用数为 0（上游重写），不要浪费时间。

---

## 5. 还能调的开关（都验证过在 0.1.59.1 里存在）

### 启动脚本里的档 B（默认关，env 可开）

```sh
STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1     # V100 FP16 tensor-core prompt 专家（改数值）
STRATA_ADAPT_OVERLAP=1                 # 拷贝与 verify 重叠（改常驻，不保证 token 一致）
STRATA_SPLIT_OWN=1                     # split 首卡自带 prompt buffer（16G 上负收益！）
STRATA_ARGMAX_ROWS_SM70=1              # 单 block 贪心 pick（仅并列 logits）
```

### 值得 A/B 的旋钮（改 `strata-chenjx-q4.json` 的 `args`，改完必须重启）

| 旋钮 | 现状 | 建议方向 |
|---|---|---|
| `--layer-split` | 25 | 试 23 / 27：GPU0 被 150 W 压着，让它少担层可能整体更快 |
| `--vram-reserve-mib` / `-later-mib` | 512 / 512 | 降到 384 换缓存（注意 512 已是临界，日志会给告警） |
| `--kv-resident` | 32768 | 降到 16384/8192，腾出的显存给专家缓存（零正确性风险，只是回读 RAM） |
| `--pcie-frac` | 0.20 | 无 NVLink，试 0.15 / 0.25 / 0.30 |
| `--pool-workers` | 14 | 18 核可试 16；配合 `STRATA_AUX_CPUS` |
| `--pipeline-windows` | 2 | 试 0（串行）对照；日志会警告两卡且 draft 不在首卡时可能更慢 |
| `--spec` | 4 | 试 6/8（改输出，属质量换速度） |
| `STRATA_PF_GROUP_SCRATCH=1` + `STRATA_PF_HC_REGION=1` + `STRATA_TQ_SUB8=1` | 未设 | fork 实测 prompt −2~−6%，但 group scratch 改舍入；符号在目标树里都有 |
| `STRATA_HC_Q8` | 0 | 开它可能省 ~0.6 GiB/卡，但要先确认 0.1.59.1 的分层上传是否已含上游 #2427 |

### 测量协议（重要，否则数字不可比）

1. 每次测量前**重启引擎**（避免 idle 预填影响），只测一次请求。
2. prompt 里加**唯一 nonce**，否则前缀缓存命中（`cached` 字段非 0）会让 prefill 时间失真。
3. 固定 `temperature=0`、固定 `max_tokens`，看返回里的 `timings.prompt_ms` / `predicted_per_second`。
4. 至少 2K / 8K / 32K 三档，各 2–3 次取中位；decode 只有 70 token 时噪声很大，要跑长一点（≥256 token）。
5. **质量回归**：用同一份 182 工具、39,845 token 的 agent 请求（模板在 `/data/models/logs/` 之外的会话记录里；简单版：构造 system + 一个"读文件"任务 + 182 个工具），确认返回的是结构化 `tool_calls` 而不是文本。

---

## 6. 下一步建议（按性价比）

1. **`--kv-resident` 下调**（16384 / 8192）→ 缓存槽增加 → 同口径三档对比。零正确性风险，最可能立刻见效。
2. **`--layer-split` 23 / 27** 对照（配合功耗墙限制）。
3. **`--pcie-frac` 三档** 对照（无 NVLink，PCIe 是瓶颈）。
4. `STRATA_PF_GROUP_SCRATCH=1` 等 trio 的 A/B（注意数值变化）。
5. 若要大收益：换一份 **sfnx 支持的量化权重**（Q4_K_XL / QAD NVFP4）后重开 split engine 路线，paskaa 在 2×V100-32G 上实测 decode 96.7→152 tok/s。

---

## 7. 仓库与文档

- 源码：`/home/chenjx/strata-v100x2-git`（0.1.59.1，已含上表改动），构建目录 `build-v100`（引擎二进制 `build-v100/strata`）、`build-split`（split 引擎，未使用）
- 参考源：`/home/chenjx/strata-v100x2-src`（旧 fork 0.1.45.7，含人工回的 prefill 偏移修复，仅对照）
- 本仓库文档：`docs/V100_PORT_CANDIDATES.md`（移植清单 + 实测数据 + 结论）、`docs/HANDOFF-V100-TUNING.md`（本文）
- 服务器上还有：`/home/chenjx/v100-changes.patch`、`v100-changes.bundle`

## 8. 其它坑

- 服务器时间 UTC；`date -u` 打日志名。
- `github.com` 直连时通时不通；机器上有全局 `url.insteadOf` 把 github.com 改写到 `gh-proxy.com`，push 会被凭据域不匹配挡住（临时 `git config --global --unset` 后再 push，用完恢复）。
- 8080 只能跑一个实例；`--engine strata` 的 serve 与 sfnx 不同进程模型。
- 构建：`cmake -S . -B build-v100 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`（`STRATA_ORCA_Q4KS_MMQ` 现在默认 ON：Q5_0 down 进 MMQ，见 `V100_PORT_CANDIDATES.md`），增量 `cmake --build build-v100 -j18`（全量约 15–20 分钟）。
