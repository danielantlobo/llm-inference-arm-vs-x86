# LLM Inference Benchmarking on HPC (llama.cpp)

Group project for the **Big Data Analysis** course (MSc in Software Engineering, University of Minho), part 2.

We built [llama.cpp](https://github.com/ggml-org/llama.cpp) natively on the **ARM and x86 nodes** of the Deucalion
supercomputer and measured how CPU inference behaves when we change the hardware, the number of threads and the
quantisation level, for three models: **Llama**, **Qwen 2.5 1.5B** and **TinyLlama**.

## What we measured

`benchmarker.py` talks to a running `llama-server` over HTTP (the model is loaded once) and records, per prompt and per trial:

| Metric | Meaning |
|---|---|
| TTFT | Time to first token (prefill latency) |
| TPOT | Time per output token (decode latency) |
| Throughput | Prefill and decode tokens per second |
| Peak memory | `VmHWM` of the server process, sampled from `/proc` |
| Goodput | Share of requests meeting an SLA (TTFT < 2 s and TPOT < 200 ms) |

Each configuration runs 3 trials per prompt (short, medium and long prompts); a summary CSV gives mean ± std.

## Experiments

1. **Threading:** 4, 8, 16 and 32 threads
2. **ARM vs x86:** same model and thread count on both partitions
3. **Quantisation:** from Q2 to Q8, trading memory for speed and quality
4. **Performance model:** predicted vs observed TPOT

Charts for each experiment are in [`graficos_relatorio/`](graficos_relatorio).

## How to run (SLURM)

```bash
sbatch compiling_arm.sh      # or compiling_x86.sh: native build with LTO
sbatch benchmarking_arm.sh   # or benchmarking_x86.sh: starts llama-server and runs benchmarker.py
python3 gerar_graficos.py    # charts from the summary CSVs
```

Paths to the models and the project folder are set at the top of the SLURM scripts.

## Layout

- `Engine_Llama/Resultados_<model>/`: raw and summary CSVs per experiment (`Threading`, `Arm_X86`, `Quantizaçao`)
- `prompts.json`: prompt set (short, medium, long)
- `gerar_grafico*.py`: plotting and performance model
