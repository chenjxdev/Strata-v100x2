# Handoff：官方 Strata 上游同步 + V100 实测配置（2026-11-21）

本文件记录把**官方上游** `Niko1221/Strata` 同步到本仓库独立分支、按官方建议参数编译、
并在 2×V100-SXM2-16GB 上实测出来的配置。

> ⚠️ **先读这一条**：本轮把 upstream 自带的 V100 opt-in 全开了（`STRATA_SM70_TABLE=1` 等），
> 但 `docs/HANDOFF-V100-TUNING.md` §2 记着 **2026-11-15 的质量否决**：
> `STRATA_SM70_TABLE=1` 实测「智商下降」，`STRATA_PF_WMMA`/`PF_WMMA2` 也已于 11-16 改回默认。
> **本轮只测了速度、没测质量**，所以提供两档配置：
> `strata-upstream-best`（速度档，全开）与 `strata-upstream-safe`（质量档，按既有结论关掉会改输出的开关）。
> 上线前请按质量档/速度档各测一轮，确认质量后再决定。

## 1. 分支与源码

| 项 | 值 |
| --- | --- |
| 上游 | `Niko1221/Strata`（官方），`upstream/main` = `3142b66b96e9c0e775e30f620b11b191a1427c2a`（2026-11-20） |
| 分支 | `upstream-sync-20261121`（本仓库），与 `upstream/main` **逐字节一致**（`git diff` 为空） |
| worktree | `/home/chenjx/strata-upstream-sync`（独立目录，不影响 `main` 上未提交的 V100 调优改动） |
| 离线备份 | `/home/chenjx/strata-upstream-sync-20261121.bundle`（104 MB，`git bundle verify` 通过） |
| 引擎版本 | `strata 0.1.63` |

取数路径：服务器到 `github.com` 直连时通时断（多次 130 s 超时），实际用
`https://gh-proxy.com/https://github.com/Niko1221/Strata.git` 走代理拉通（~50 KB/s）。

**该分支尚未推到 GitHub**：push 到了 GitHub 但被拒 —— PAT 缺 `workflow` scope，
而新分支多出 `.github/workflows/backends.yml`。换带 `workflow` 权限的 token 即可原样推：

```sh
cd ~/strata-v100x2-git
git push https://<user>:<PAT-with-workflow-scope>@gh-proxy.com/https://github.com/chenjxdev/Strata-v100x2.git \
  upstream-sync-20261121:refs/heads/upstream-sync-20261121
```

## 2. 编译（官方建议参数）

出处：upstream `docs/NVIDIA_V100.md`、`docs/SOURCE_INSTALL.md`、`docs/OLDER_GPUS.md`（CUDA 12 引擎段）。

```sh
cmake -S engine -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build --target strata -j 24
```

- `-S engine` 是 0.2.16+ 的新目录布局；**新代码里已经没有 llama.cpp**（`engine/llama.cpp` 不存在，
  `GGML_*`/`LLAMA_*` 选项数为 0），因此全部用新的 `STRATA_*` 选项
- CUDA 12.9（CUDA 13 砍掉 Volta）；CUDA 12 需要较新的 gcc（本机 13.3 可用）
- `STRATA_EXPERIMENTAL_SM60=ON` 是官方 V100 页明确要求的：把 Volta 的 prompt attention、
  #540 attention kernel、Pascal BF16 路径编进来，并把运行时门槛降到 CC 6.0
- 旧 fork 当初用的五个选项与上面**完全相同**（`build-v100/CMakeCache.txt` 可查），
  所以下面的 A/B 差的是代码而不是编译参数

## 3. 实测（同一模型 / 同一 layer_split / 同一缓存槽位）

模型 `/data/models/packs/ud-iq4_xs-abliterated`（UD-IQ4_XS），`--max-context 262144 --kv int8
--spec 4 --mtp --pipeline-windows 2`，2×V100-SXM2-16GB。4 个 prompt（code / copy / prose / long 2108 tok）× 5 轮，
temperature 0，`max_tokens 256`。

| 组 | 变化 | 总体中位 | 均值 | warm 中位 |
| --- | --- | --- | --- | --- |
| 官方基线 | upstream 默认 | 87.75 | 85.75 | — |
| 官方 +opt-in | 7 个 V100 opt-in 全开 | 90.00 | 90.38 | — |
| D | +memlock +`--pcie-frac 0.20` | 61.15 | 61.21 | 67.65 |
| **E** | **+memlock（pcie 让上游自动选）** | **98.35** | **97.09** | **99.05** |
| F | +memlock +`layer_split 32` | 46.35 | 46.84 | 47.80 |
| 对照：旧 fork 0.1.60.1 | fork 生产配置 | 81.70 | 84.10 | — |

三条结论：

1. **memlock 是最大的一块**（+9~14%）：引擎要锁 26.8 GiB 的 PLE 表，本机 `RLIMIT_MEMLOCK` 只有 15.7 GiB，
   引擎日志自己说 `the lock was measured up to 6% faster in decode (#4671)`。
   抬到 unlimited 后日志变成 `PLE table locked in RAM (--ple-io ram)`。
2. **`--pcie-frac 0.20` 有害**（旧 fork 的值，塌 37%）：upstream 自己探 PCIe 得 0.15，就让它自动选。
   注意 0.1.63 的 `--help` 里已经没有这个 flag，但引擎仍然接受。
3. **`layer_split 32` 有害**（旧 fork 当年的偏好，只剩 46 tok/s）：新代码会自己探 PCIe 再分专家缓存，
   25 的缓存槽是 4594，32 只有 4290。**保持 25。**

另外：`STRATA_PF_WMMA` / `PF_WMMA2` / `Q8_GATHER_WIDE=0` / `GDN_REC_WARP=2` 这组 prompt 侧开关在本机
**没有速度收益**（2108 tok 的冷 prefill：761 / 761 / 755 tok/s 三臂一致），
但 upstream 文档写明 `PF_WMMA=1` 会用 FP16 激活替代 MMQ 的 q8_1、`GDN_REC_WARP=2` 会改 FP32 求和顺序
—— **这两项会改输出**。

## 4. 正式脚本与两档配置

| 文件 | 说明 |
| --- | --- |
| `tools/v100/start-strata-upstream.sh` | 启动/停止/查状态。服务器上装在 `/data/models/start-strata-upstream.sh` |
| `tools/v100/strata-upstream-fast.json` | 速度档（= 服务器上的 `strata-upstream-best.json`），7 个 opt-in 全开 |
| `tools/v100/strata-upstream-safe.json` | 质量档（= 服务器上的 `strata-upstream-safe.json`），关掉会改输出的开关 |
| `tools/v100/strata-upstream.service` | systemd unit 样例（`LimitMEMLOCK=infinity`），默认不安装 |

```sh
/data/models/start-strata-upstream.sh start strata-upstream-best   # 速度档
/data/models/start-strata-upstream.sh start strata-upstream-safe   # 质量档
/data/models/start-strata-upstream.sh status
/data/models/start-strata-upstream.sh stop
```

两档共用：`layer_split 25`、`gpu [1,0]`、port 8080、`--pcie-frac` 不写、`--vram-reserve-*` 不写
（0.1.63 已删，官方换成 `--vram-elastic` / `--vram-segment-mib`）、`STRATA_PREFILL_PROMOTE` 不写
（fork 专属，官方二进制没有这个符号）。

**memlock 必须抬**，否则 PLE 表锁不住、白扔 6~10%：脚本会尝试 `sudo -n prlimit --memlock=unlimited --pid $$`，
不行就打印告警（需要 root 抬 hard limit，或用 systemd 的 `LimitMEMLOCK=infinity`）。

## 5. 踩过的坑

1. **`pkill -f` 会杀掉自己**：`sshrun.py` 是把整个脚本当远程命令行发过去的，脚本里出现的
   `pkill -f 'strata-upstream-sync/build/strata'` 会匹配到执行脚本的 shell 自己，SSH 直接断（exit -1）。
   凡是 `pkill/pgrep -f` 的模式串一律用 `[x]` 打断，并且按 PID kill。正式脚本是**文件**形式运行，不受此影响。
2. **GitHub 直连时通时断**：`git fetch origin` 可能 130 s 超时，用 `gh-proxy.com` 兜底。
3. **0.1.63 删了 `--vram-reserve-mib` / `--vram-reserve-later-mib`**，旧配置原样搬会起不来。
4. **`--pcie-frac` 从 `--help` 消失但引擎仍接受**（启动日志里会提示）。
5. **官方 upstream 没有 V100 默认开关表**：`engine/src/core/nv_defaults.cpp` 只有 sm_86 / sm_120 / sm_121 / sm_60，
   `cc_major < 8` 直接走 FP16 prompt 老路 —— V100 是 upstream 里最没被调过的路径，所有 V100 优化默认全关。

## 6. 还没试的候选

1. **`STRATA_MTP_STAGE0=1`**：引擎日志提示两段切分下草稿层不在首卡时 `--pipeline-windows 2`
   在别的机器上「lost to the serial loop」，要靠这个开关把草稿层放到首卡。
2. **`--vram-elastic` / `--vram-segment-mib`**：upstream 替掉旧 reserve 机制；引擎还打印
   `--kv-grow is off (... no --vram-elastic ...)`，也就是配置里的 `STRATA_KV_GROW=1` 目前被层切分判成无效。
3. **三个 decode opt-in 拆开单测**（这次是打包测的）。
