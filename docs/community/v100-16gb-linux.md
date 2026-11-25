# One V100 16 GB on Linux: a user's report

Reported by **[ildargiv](https://github.com/ildargiv)** in November 2026, in the fork
[ildargiv/lapilli-v100](https://github.com/ildargiv/lapilli-v100) ([full notes](https://github.com/ildargiv/lapilli-v100/blob/main/docs/v100-16gb-linux.md),
raw results in its `results/v100-16gb-linux/`). It is the first run of Lapilli on a **16 GB V100**, a card this README listed
as not tested, with the configuration that makes it stable.

## Thank you, ildargiv

ildargiv ran Lapilli on a card half the size of the one it was written for, hit the out-of-memory crash that a 16 GB card
meets on long prompts, found its cause and the two flags that fix it, then measured every option one at a time, four runs
each, against the engine already in use there, and published the raw data, the protocol and a report template for others. Every
V100 16 GB owner can now start from a configuration that works. Thank you for this careful and generous work.

The figures below are copied as reported (medians of four runs, measured by the client while streaming), with Lapilli v0.4.

## The machine

| | |
|---|---|
| GPU | 1 x Tesla V100-SXM2 16 GB |
| CPU | AMD Ryzen 5 5600G (AVX2, no AVX-512) |
| System RAM | 96 GiB |
| System | Ubuntu 22.04, CUDA 12.9 |
| Pack | `Qwen3.8-FN-UD-IQ3_XXS.basalt` from `jesdga/Qwen3.8-Flash-Next-Basalt` (87.6 GB) |

## The stable 16 GB configuration

With the default reserve, the automatic expert cache grew to about 9.3 to 9.6 GiB and prompts of 8K and 32K tokens failed
with `mmq: out of memory`, then the server restarted in a loop. Two flags fix it:

```bash
bin/basalt serve Qwen3.8-FN-UD-IQ3_XXS.basalt --gpu 0 --kv int8 --kv-resident 32768 --max-context 262144 \
  --mmap-experts --resident-experts --ram-reserve-mib 10240 --vram-reserve-mib 700 --port 8080
```

At load: 13.5 GiB of weights on the GPU and about 37.7 GiB in system RAM, an expert cache of 4,602 slots (8.82 GiB).
`--kv k8v4` does not work with `--kv-resident` in this build.

## Results

Writing 256 tokens after a prompt of 886, 6,214 and 24,359 tokens, with the stable configuration:

| | 1K prompt | 8K prompt | 32K prompt |
|---|---|---|---|
| Writing | 67.8 tokens/s | 76.0 tokens/s | 70.4 tokens/s |
| Prompt reading | 497 tokens/s | 1,033 tokens/s | 1,174 tokens/s |
| Time to first token | 1.78 s | 6.02 s | 20.75 s |

Against Strata 0.1.60 on the same machine (a different pack, `GSQ-RCO IQ3_S` in GGUF, with its drafter; one run, so not a
like-for-like comparison, as the report notes):

| | Strata, writing | Lapilli, writing | Strata, prompt reading | Lapilli, prompt reading |
|---|---|---|---|---|
| 1K prompt | 91.4 | 67.8 | 410 | 497 |
| 8K prompt | 89.3 | 76.0 | 870 | 1,033 |
| 32K prompt | 94.6 | 70.4 | 1,213 | 1,174 |

ildargiv's reading: on 16 GB, Lapilli's writing is limited by its expert cache, which finds about 73% of the experts it needs in
VRAM against 97 to 99% for Strata with a cache of the same size; the explicit adaptive flags did not change it, and the lossy
routing flags made prompts slower. The report also lists two Volta fallbacks seen in the log (`mid_chain` and `dense_tc_pair`).
The notes give every experiment, A to G.
