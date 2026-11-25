# One V100 PCIe on Linux: a user's report

Reported by **u/nebod99** on Reddit, November 2026, a few hours after this fork was published. It is the first run of
Lapilli outside the machine it was developed on, and it covers three things the port had not tried: **Linux**, a **single
V100**, and a **PCIe card**. A second report followed, on the [quality](#quality) of the answers.

## Thank you, nebod99

Before this report, Lapilli had only ever run on one machine. nebod99 built it on a platform it had never seen, ran it on a
single card next to the engine already in use there, measured both the same way, and wrote down everything someone else
needs to do the same: the build flags, the Docker image, the pack revision and how to convert it. Thanks to that work, every
V100 owner now knows the port runs on Linux, on one card and on PCIe, at about 100 tokens/s. nebod99 then came back with a
second report: more than 2,500 questions, to check that the speed costs nothing in the quality of the answers. It does not.

Thank you, sincerely, for the time, the care and the generosity of sharing it. This page exists because of you.

The figures below are copied as reported. They are initial measurements, taken from each engine's own timing output.

## The machine

| | |
|---|---|
| GPU | 1 x Tesla V100 PCIe 32 GB |
| CPU | Intel Core i9-10940X |
| System RAM | 96 GB |
| OS | Linux, in a CUDA 12.6.3 Docker image |
| Pack | Qwen3.8-Flash-Next GSQ-RCO `IQ3_S` (revision v1, converted locally to v2) |
| Lapilli build | commit `7588701`, `-DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp` |
| Lapilli options | int8 KV cache, memory-mapped and resident experts |

## The test

Thinking enabled. The long prompt is 7,242 input tokens, and every run generates 1,024 tokens. The long prompt fills about
7.2K tokens of the context window.

| Engine, configured context | Decode, short prompt | Decode, long prompt | First token, long prompt | Launch to ready |
|---|---:|---:|---:|---:|
| Strata, 128K | 57.7 tokens/s | 55.6 tokens/s | 22.6 s | 70 s |
| **Lapilli, 16K** | **112.2 tokens/s** | **107.3 tokens/s** | **8.8 s** | 124 s |
| **Lapilli, 128K** | **99.2 tokens/s** | **100.9 tokens/s** | **8.7 s** | 90 s |

## What it shows

- **One V100 is enough.** The full Qwen3.8-Flash-Next runs at about 100 tokens/s on a single PCIe card, with a 128K context.
- **The first token comes 2.6 times sooner** than with Strata on the same machine: under 9 seconds instead of about 23, once
  the model is loaded.
- **Long-prompt decode is about 1.8 times faster** than Strata at the same configured context.
- **Startup is slower than Strata's** on this machine (90 to 124 s against 70 s). It is paid once per launch. Launch time
  depends a lot on the disk and on what the OS already has in its file cache, so the order of the runs matters here.

## Quality

nebod99's second report checks the answers themselves. Every local model on that machine goes through the same personal
test suite: questions selected from public benchmarks, not the full benchmarks. This run is Lapilli on the machine and pack
above, with thinking on, no thinking budget and temperature 0.

In nebod99's words, the results "essentially tie the same model on Strata", and they are "tied for best test results I
have ever achieved by any model / backend". The difference is speed: about 100 tokens/s on Lapilli, against 55 on Strata.

The run, as printed by the test tool:

```text
Model: qwen3.8-flash-next-iq3_s-lapilli (PRODUCTION)
Suite: 983 questions | Thinking: ON | Budget: None | Backend: lapilli| Temperature: 0
```

| Set | Test | Passed | Questions |
|---|---|---:|---:|
| First set | `ifeval` | 90.50% | 200 |
| | `math_500` | 99.00% | 200 |
| | `mmlu_pro` | 86.73% | 196 |
| | `bbh_causal` | 68.45% | 187 |
| | `mbpp_plus` | 84.00% | 200 |
| Hard set | `ifbench` | 71.33% | 150 |
| | `zebralogicbench` | 95.33% | 150 |
| | `musr-murder` | 79.10% | 134 |
| | `musr-objects` | 78.20% | 133 |
| | `musr-teams` | 81.20% | 133 |
| | `acebench` | 80.71% | 850 |

The first set is the suite's 983 questions; the hard set adds 1,550 cases. Strata's scores were not posted, only that the
two engines tie.

This agrees with the port's own check on its development machine, with Basalt's quality gate and the `UD-Q4_K_XL` pack:
the same perplexity and KLD as Basalt's published figures (see [What was measured](../../README.md#what-was-measured)).

## Building it on Linux

The port was only developed on Windows; this report shows that the same tree builds and runs on Linux without changes:

```sh
cmake -S . -B build70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp
cmake --build build70
```

with CUDA 12.6 (here the 12.6.3 Docker image). A pack published in Basalt's v1 format is converted once with Basalt's own
tool, `tools/migrations/basalt_v1_to_v2.py` (see [tools/migrations/README.md](../../tools/migrations/README.md)).

Back to the [README](../../README.md).
