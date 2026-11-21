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

## 6. 重要差异：prompt 路径没有 MMQ（上游 0.2.16 移除了它）

本仓库的 11-16 审计记录写过：**`STRATA_MMQ` 在上游默认 OFF，不带 `-DSTRATA_MMQ=ON` 重建会掉约 30% prefill**
（14K prompt：1040-1091 vs 1595-1621 tok/s）。这条结论今天仍然成立，但原因要说清楚：

- **上游 0.1.63 的新 `engine/` 树里根本没有 `STRATA_MMQ` 这个构建选项。**
  它的 cmake 里只剩 `STRATA_DENSE_MMQ` 和 `STRATA_MMQ_DIRECT` 两个 MMQ 相关变量。
  upstream 的 `docs/DETAILS.md` 写得很直接：「The release builds have no MMQ」——
  那套 prompt 专家的 MMQ kernel 是从 llama.cpp 派生的，upstream 在 0.2.16 整条移除。
- **fork 仍然有**：`CMakeLists.txt:112` 保留
  `option(STRATA_MMQ "Build llama.cpp's MMQ prompt kernels (CUDA; HIP together with STRATA_PREFILL_MMQ)" OFF)`，
  加上 `STRATA_MMQ_KQUANTS=ON`、`STRATA_MMQ_TQ_OVERLAY=ON`（后者要 `third_party/strata-ggml` 的 ggml-cuda overlay）。
  fork 的 `build-v100/CMakeCache.txt` 里 `STRATA_MMQ:BOOL=ON`。
- **所以在上游 0.1.63 上加 `-DSTRATA_MMQ=ON` 没有意义**（选项不存在），想拿回 MMQ 的 prefill 速度
  要把 fork 那套 llama.cpp overlay + MMQ prompt kernel **移植到新 engine 布局** —— 那是代码移植，不是配置项。
- **上游给的新路是 fused int8 prompt experts（`STRATA_PF_FUSED=1`），但在 V100 上用不了**（2026-11-21 实测）：
  - 实测：17~18K token 的 prompt，两臂（不带 / 带 `STRATA_PF_FUSED=1`）prefill
    **1656.6 / 1677.2 / 1697.8 vs 1663.5 / 1670.6 / 1697.2 tok/s** —— 差 -0.4%，纯噪声。
  - 原因在源码里写死了：`engine/src/qwen35/q36_engine.cu:1418`
    「... cards under 40 SMs and **compute capability below 8 are as before (MMQ)**」——
    fused int8 prompt 专家只给 **CC ≥ 8**（sm_80+）的卡用，**V100 是 CC 7.0，开关被直接忽略**（引擎也不报警）。
  - 也就是说：**在 V100 上，上游 0.1.63 既没有 MMQ、也没有 fused int8 专家**，
    prompt 走的是分组 kernel / FP16 路径，这是上游的现状，不是配置能改的。
  - 本轮实测的 `STRATA_PF_WMMA=1`（V100 官方页推荐的 FP16 tensor-core prompt 专家）**也没有 prefill 收益**
    （2108 tok 冷 prefill：761 / 761 / 755 tok/s 三臂一致）。
  - 附带一个值得注意的数据点：上游构建（**完全没有 MMQ**）在 17~18K token 上跑到 **~1670 tok/s**，
    而本仓库 11-16 审计记录里的 fork 数据是「无 MMQ 1040-1091 / 有 MMQ 1595-1621 tok/s（14K）」——
    **上游这版的 prompt 吞吐已经不在 fork 之下**（口径/prompt 不同，要严格比较需同 prompt A/B）。

**2026-11-21 实测：MMQ 在这台机器上没有收益**（同包、同 prompt，只差二进制）：

| 二进制 | 模型包 | 17~18K prompt 冷 prefill 中位 |
| --- | --- | --- |
| fork `build-v100/strata`（MMQ=ON） | 官方三 shard 包 | **1682.4 tok/s** |
| 上游 `strata-upstream-sync/build/strata`（无 MMQ） | 官方三 shard 包 | **1677.2 tok/s** |
| （参考）fork 生产配置 + UD 四 shard 包 | UD-IQ4_XS | 1693.8 tok/s |

差 +0.3%，噪声内。

**2026-11-21 晚：MMQ 真的移植进上游并做了同二进制 A/B**（不再只是跨二进制比较）：

- 移植范围：`moe_mmq.cu` / `moe_iq4_w4a8.cu` / `ggml_cuda_host.cu` / `mmq_nvfp4_w4a8.cu` 从 fork 搬进 `engine/src/prefill/`，
  加 `strata_ggml_mmq_types()` + 精简版 `strata_cuda_mmq()`，打开 `-DSTRATA_MMQ=ON`（9 个 MMQ 实例：
  q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0；不带 trellis/TAB/K-quant DSO/ORCA；
  NVFP4 W4A4 单元需要 sm_120a，用空桩）。产物 `build-mmq/strata`（119 MB，比无 MMQ 的 90 MB 多 29 MB）。
- 运行确认 MMQ 真的在跑：`strings` 里 1687 个 mmq 字样（无 MMQ 版 272），引擎日志出现 MMQ 独有的
  `strata serve: the prompt path borrows 2308 CUDA0 cache slots (4.79 GiB)`。
- 同配置、同包、同 prompt，只差二进制的 A/B：

| 指标 | MMQ ON | MMQ OFF | 差 |
| --- | --- | --- | --- |
| prefill 17~18K 中位 | 1674.7 tok/s | 1676.2 tok/s | -0.1% |
| prefill 均值 | 1673.4 | 1669.3 | +0.2% |
| decode 中位 | 96.75 | 97.65 | -0.9% |
| decode 均值 | 96.14 | 96.14 | 0 |
| decode warm 中位 | 103.15 | 99.40 | +3.8%（噪声） |

**结论：即使把 MMQ 移植进来，在这台 V100 + UD/官方 IQ4_XS 包上也没有收益。**
移植代码留在 `/home/chenjx/strata-mmq-port.patch` + `strata-mmq-port-README.txt`（未应用到分支），
工作树已还原为与 upstream 一致。

**11-16 的「不带 `-DSTRATA_MMQ=ON` 重建掉 30% prefill」在当前上游代码/该包上不复现**
（那次是 14K prompt + 当时的代码；上游这版把 FP16 prompt 路径做上来了）。
结论：**不要把 fork 的 MMQ 路径移植到新 engine** —— 上游 `cmake/backends/cuda.cmake` 还专门加了
`tools/check_no_mmq.py` 守门测试（二进制里出现 MMQ 符号即失败），移植等于跟上游的显式移除对着干，
而收益为零。除非换到 MMQ 确实更快的负载（K-quant 专家 / 其它芯片）再评估。

decode 侧本轮实测也是上游 ≈ fork（见 §3）。

## 7. V100 上还没试过的候选：逐个试过了

| 候选 | 结果 | 机制 |
| --- | --- | --- |
| `STRATA_MTP_STAGE0=1` | **负** | 草稿层挪到首卡后首卡多占 220 MiB，专家缓存从 4594 掉到 4176 槽（-9%），命中率下降 |
| `vm.nr_hugepages=28689`（大页） | **别做** | 见下节：把机器搞挂 |
| `--vram-elastic` | 不是提速旋钮 | upstream 的定义是「专家缓存分段，让 `POST /v1/vram` 能在请求之间把显存还给别的程序」——是显存共享功能；它会顺带让 `STRATA_KV_GROW` 生效 |

### 7.1 `STRATA_MTP_STAGE0=1` 实测（同配置，唯一变量）

| 指标 | base | STAGE0=1 | 差 |
| --- | --- | --- | --- |
| prefill 17~18K 中位 | 1677.2 tok/s | 1647.8 | -1.8% |
| prefill 均值 | 1671.4 | 1646.2 | -1.5% |
| decode 中位 | 94.50 | 89.90 | -4.9% |
| decode 均值 | 93.02 | 90.56 | -2.6% |
| decode warm 中位 | 96.45 | 92.15 | -4.5% |

引擎日志说明了原因：`STRATA_MTP_STAGE0=1` 让草稿层落到首卡（`+220 MiB for the draft head`），
首卡的专家缓存从 4594 槽降到 4176 槽（-9%），命中率下降 → decode 变慢。
（`long` prompt 上 STAGE0 反而快 7.7%，两个方向都有，属噪声带内，但整体和机制一致为负。）**保持关闭。**

### 7.2 教训：不要在这台机器上预留大页（2026-11-21 12:30 事故）

试 `vm.nr_hugepages=28689`（= 56 GiB）想给专家 arena 用 MAP_HUGETLB，结果：

- 引擎直接拒绝：`MAP_HUGETLB unavailable (needed 28689 2 MiB pages, vm.nr_hugepages=28689);
  transparent huge pages NOT requested: only 0 GiB of free 2 MiB blocks for a 56 GiB arena,
  compaction would stall the load (STRATA_ARENA_THP=1 requests them anyway) (#2126)`
- 预留之后内核进入内存规整/回收：journald 连续 `Under memory pressure, flushing caches`，
  `workqueue: fill_page_cache_func hogged CPU`（12:31 起）
- 12:37:14 `systemd-networkd: oray_vnc: Link DOWN` —— 蒲公英 VPN 掉线，机器随之失联
- **没有 OOM-kill 事件**（不是被 OOM 杀的，是内存压力 + VPN 断链），12:47 机器重启

结论：引擎自己都写了「compaction would stall the load」。这台机器在这个负载下给不出 56 GiB 连续大页，
再试只会重复把机器搞挂。**大页这条路在 V100 上放弃**（THP 的 madvise 模式已经在用，是默认路径）。

## 8. 还没试的候选

0. ~~**`STRATA_PF_FUSED=1`**~~：已测，**V100 上无效**（CC 7.0 被门槛挡掉，见 §6）。
1. ~~同 prompt 的上游 vs fork prefill 对比~~：**已测，1682.4 vs 1677.2 tok/s，MMQ 无收益**（见 §6）。
2. **`STRATA_MTP_STAGE0=1`**：引擎日志提示两段切分下草稿层不在首卡时 `--pipeline-windows 2`
   在别的机器上「lost to the serial loop」，要靠这个开关把草稿层放到首卡。
3. **`--vram-elastic` / `--vram-segment-mib`**：upstream 替掉旧 reserve 机制；引擎还打印
   `--kv-grow is off (... no --vram-elastic ...)`，也就是配置里的 `STRATA_KV_GROW=1` 目前被层切分判成无效。
4. **三个 decode opt-in 拆开单测**（这次是打包测的）。
