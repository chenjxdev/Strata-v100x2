# Inkling 2 Mini on V100: a port built on Lapilli

Built by **[Csavoldi](https://github.com/Csavoldi)** in November 2026, in the fork
[Csavoldi/lapilli-inkling](https://github.com/Csavoldi/lapilli-inkling) (branch `inkling2-port`, notes in its
[INKLING2.md](https://github.com/Csavoldi/lapilli-inkling/blob/inkling2-port/INKLING2.md)). Inkling 2 Mini, by Thinking
Machines Lab, is a looped mixture-of-experts model (`inkling2_loop`, 26.2B parameters, 3.4B active) that upstream Basalt added
in v0.5.5, for Blackwell only.

## Thank you, Csavoldi

Csavoldi merged upstream Basalt v0.5.5 onto Lapilli, resolved its thirteen conflicts by hand so that every Volta path stays
alive, made the Inkling 2 kernels run on sm_70 (attention without clusters, `ldmatrix`, `mma` and `cp.async` through
Lapilli's Volta layer), and checked the result against llama.cpp before measuring it. A new model now runs on a V100 that its
authors never targeted. Thank you for taking the port somewhere new, and for documenting every step.

The figures below are copied as reported (validated on November 21, 2026).

## The machine and the model

| | |
|---|---|
| GPUs | 2 x V100 PCIe 32 GB, one in use |
| Weights | the official release: experts in NVFP4, unchanged; dense layers MXFP8 to Q8_0; embedding in BF16, in host RAM |
| VRAM | about 16 GB of weights, which leaves room for the cache on one card |

## Results

| | |
|---|---|
| Logits against the llama.cpp reference | 5 of 5 top-1 matches, KL 0.0001 to 0.0005, correlation 0.9997 or more |
| Writing | 69 tokens/s with an empty context, 60.5 at 42K, 44.4 at 165K |
| Prompt reading | 850 tokens/s on a short prompt, 439 at 42K, 211 at 165K |
| Context | 512K configured and measured (the model's own is 1M) |

The known gaps are all about speed, not correctness: the DSpark drafter needs kernels that exist only for sm_120a (the
numbers above are without a drafter); prompt reading of NVFP4 experts goes through the emulated `mma` path; `--kv fp8` runs as
fp16 on Volta. Lapilli v0.5's NVFP4 prompt products on Volta's tensor cores may help with the second.

To build it: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=inkling2`, CUDA 12.6 or newer (12.9 tested), and
convert the official checkpoint with `tools/convert/convert.py --checkpoint DIR --out MODEL.basalt`. See the fork for the
details.
