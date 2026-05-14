#!/usr/bin/env python3
"""
The model is loaded ONCE by the server — no reloading between prompts.

Metrics collected per trial:
  1. TTFT        — prompt_ms from server timings
  2. TPOT        — predicted_per_token_ms from server timings
  3. Throughput  — prefill (prompt_per_second) + decode (predicted_per_second)
  4. Peak Memory — VmHWM from /proc/<server_pid>/status (sampled every 2s)
  5. Goodput     — fraction of requests meeting SLA (TTFT < 2000ms AND TPOT < 200ms)

After all trials, a summary CSV is written with mean ± std for each metric per prompt.
"""

import argparse
import csv
import json
import math
import os
import threading
import time
import urllib.request

# Configuração
MAX_NEW_TOKENS       = 256
PROMPTS_FILE         = "prompts.json"
# Ajustado para x86 em vez de arm para refletir a tua partição atual[cite: 3]
RAW_CSV              = "results/arm2/metrics/raw_benchmarks.csv"
SUMMARY_CSV          = "results/arm2/metrics/summary_benchmarks.csv"
MANDATORY_IDS        = {"S1_MANDATORY", "M1_MANDATORY", "L1_MANDATORY"}
MANDATORY_OUTPUT_DIR = "results/arm2/outputs"

SLA_TTFT_MS = 2000   # ms — max acceptable time to first token
SLA_TPOT_MS = 200    # ms — max acceptable time per output token

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server inference on HPC nodes via HTTP."
    )
    parser.add_argument("--threads",    type=int, required=True,
        help="Number of CPU inference threads (informational).")
    parser.add_argument("--model",      type=str, required=True,
        help="Path to GGUF model (informational; logged to output).")
    parser.add_argument("--server-url", type=str, default="http://localhost:8080",
        help="Base URL of the running llama-server.")
    parser.add_argument("--server-pid", type=int, default=None,
        help="PID of the llama-server process (for peak memory tracking).")
    parser.add_argument("--trials",     type=int, default=3,
        help="Number of trials per prompt (default: 3).")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate per request (default: {MAX_NEW_TOKENS}).")
    return parser.parse_args()


# Load de prompts
def load_prompts(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_prompts = []
    for category, prompt_list in data["prompts"].items():
        for p in prompt_list:
            p["category"] = category
            flat_prompts.append(p)

    print(f"[INFO] Loaded {len(flat_prompts)} prompts from '{filepath}'.")
    return flat_prompts

# Média (dos 3 trials)
def mean(values: list):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)

# Desvio padrão (dos 3 trials)
def stdev(values: list):
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n < 2:
        return 0.0
    mu  = sum(clean) / n
    var = sum((x - mu) ** 2 for x in clean) / (n - 1)
    return round(math.sqrt(var), 4)

# Peak Memory
class MemorySampler:
    def __init__(self, pid: int):
        self.pid     = pid
        self.peak_mb = 0.0
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._read_vmhwm())
        return self.peak_mb

    def _read_vmhwm(self) -> float:
        try:
            with open(f"/proc/{self.pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return round(int(line.split()[1]) / 1024, 2)
        except Exception:
            pass
        return 0.0

    def _run(self):
        while not self._stop.is_set():
            val = self._read_vmhwm()
            if val > self.peak_mb:
                self.peak_mb = val
            self._stop.wait(2)

# Server health check 
def wait_for_server(base_url: str, timeout: int = 180):
    health_url = f"{base_url}/health"
    deadline   = time.time() + timeout
    print(f"[INFO] Waiting for server at {health_url} ...", flush=True)

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if json.loads(resp.read().decode()).get("status") == "ok":
                    print("[INFO] Server is ready.")
                    return
        except Exception:
            pass
        time.sleep(2)

    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout}s.")

# Inferência (HTTP POST)
def run_inference(prompt_text: str, server_url: str, max_tokens: int) -> dict:
    endpoint = f"{server_url}/completion"

    payload = json.dumps({
        "prompt":      f"Question: {prompt_text}\nAnswer: ",
        "n_predict":   max_tokens,
        "temperature": 0.7,
        "top_p":       0.9,
        "stream":      False,
        "cache_prompt": False
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    t_start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        response_body = json.loads(resp.read().decode("utf-8"))
    wall_time_s = round(time.perf_counter() - t_start, 4)

    timings = response_body.get("timings", {})

    ttft_ms            = timings.get("prompt_ms")
    tpot_ms            = timings.get("predicted_per_token_ms")
    throughput_prefill = timings.get("prompt_per_second")
    throughput_decode  = timings.get("predicted_per_second")
    output_text        = response_body.get("content", "").strip()

    # SLA check
    meets_sla = (
        ttft_ms is not None and ttft_ms < SLA_TTFT_MS and
        tpot_ms is not None and tpot_ms < SLA_TPOT_MS
    )

    return {
        "ttft_ms":            ttft_ms,
        "tpot_ms":            tpot_ms,
        "throughput_prefill": throughput_prefill,
        "throughput_decode":  throughput_decode,
        "wall_time_s":        wall_time_s,
        "meets_sla":          meets_sla,
        "output_text":        output_text,
    }

# Output helpers
def save_mandatory_output(prompt_id: str, trial: int, output_text: str):
    os.makedirs(MANDATORY_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(MANDATORY_OUTPUT_DIR, f"{prompt_id}_trial{trial}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"    [SAVED] Output → {path}")

RAW_FIELDS = [
    "prompt_id", "category", "trial",
    "ttft_ms", "tpot_ms",
    "throughput_prefill_tps", "throughput_decode_tps",
    "wall_time_s", "meets_sla", "peak_memory_mb"
]

SUMMARY_FIELDS = [
    "prompt_id", "category",
    "ttft_mean_ms",     "ttft_std_ms",
    "tpot_mean_ms",     "tpot_std_ms",
    "prefill_mean_tps", "prefill_std_tps",
    "decode_mean_tps",  "decode_std_tps",
    "wall_mean_s",      "wall_std_s",
    "peak_memory_mb",
    "sla_pass_count", "sla_total_count", "goodput_pct"
]

def initialise_csv(filepath: str, fields: list):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    print(f"[INFO] Initialised CSV → '{filepath}'")

def append_csv_row(filepath: str, fields: list, row: dict):
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)

# Summary Writer
def write_summary(all_rows: list, peak_mb: float):
    groups = {}
    for row in all_rows:
        pid = row["prompt_id"]
        if pid not in groups:
            groups[pid] = {"category": row["category"], "trials": []}
        groups[pid]["trials"].append(row)

    print(f"\n[INFO] Writing summary → '{SUMMARY_CSV}'")
    print(f"\n{'─'*95}")
    print(f"{'Prompt':<20} {'TTFT mean±std (ms)':<24} {'TPOT mean±std (ms)':<24} "
          f"{'Decode tps mean±std':<24} {'Goodput':>8}")
    print(f"{'─'*95}")

    for prompt_id, data in groups.items():
        trials   = data["trials"]
        category = data["category"]

        ttft_vals  = [t["ttft_ms"]            for t in trials]
        tpot_vals  = [t["tpot_ms"]            for t in trials]
        pre_vals   = [t["throughput_prefill"]  for t in trials]
        dec_vals   = [t["throughput_decode"]   for t in trials]
        wall_vals  = [t["wall_time_s"]          for t in trials]
        sla_passed = sum(1 for t in trials if t["meets_sla"] is True)
        sla_total  = len([t for t in trials if t["meets_sla"] is not None])
        goodput_pct = round((sla_passed / sla_total) * 100, 1) if sla_total > 0 else 0.0

        summary_row = {
            "prompt_id":        prompt_id,
            "category":         category,
            "ttft_mean_ms":     mean(ttft_vals),
            "ttft_std_ms":      stdev(ttft_vals),
            "tpot_mean_ms":     mean(tpot_vals),
            "tpot_std_ms":      stdev(tpot_vals),
            "prefill_mean_tps": mean(pre_vals),
            "prefill_std_tps":  stdev(pre_vals),
            "decode_mean_tps":  mean(dec_vals),
            "decode_std_tps":   stdev(dec_vals),
            "wall_mean_s":      mean(wall_vals),
            "wall_std_s":       stdev(wall_vals),
            "peak_memory_mb":   peak_mb,
            "sla_pass_count":   sla_passed,
            "sla_total_count":  sla_total,
            "goodput_pct":      goodput_pct,
        }
        append_csv_row(SUMMARY_CSV, SUMMARY_FIELDS, summary_row)

        ttft_str = f"{summary_row['ttft_mean_ms']} ± {summary_row['ttft_std_ms']}"
        tpot_str = f"{summary_row['tpot_mean_ms']} ± {summary_row['tpot_std_ms']}"
        dec_str  = f"{summary_row['decode_mean_tps']} ± {summary_row['decode_std_tps']}"
        print(f"{prompt_id:<20} {ttft_str:<24} {tpot_str:<24} {dec_str:<24} {goodput_pct:>7}%")

    print(f"{'─'*95}")

# Goodput summary
def print_goodput_summary(all_rows: list):
    total   = len([r for r in all_rows if r["meets_sla"] is not None])
    passed  = len([r for r in all_rows if r["meets_sla"] is True])
    goodput = round((passed / total) * 100, 1) if total > 0 else 0.0

    categories = {}
    for r in all_rows:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        if r["meets_sla"] is not None:
            categories[cat]["total"]  += 1
            categories[cat]["passed"] += int(r["meets_sla"] is True)

    print("\n" + "=" * 60)
    print(f"  GOODPUT SUMMARY  (SLA: TTFT<{SLA_TTFT_MS}ms, TPOT<{SLA_TPOT_MS}ms)")
    print("=" * 60)
    print(f"  Overall : {passed}/{total} requests passed ({goodput}%)")
    for cat, c in categories.items():
        pct = round((c["passed"] / c["total"]) * 100, 1) if c["total"] > 0 else 0
        print(f"  {cat:<10}: {c['passed']}/{c['total']} ({pct}%)")
    print("=" * 60)

# Main benchmark loop
def run_benchmark(args):
    wait_for_server(args.server_url)
    prompts = load_prompts(PROMPTS_FILE)
    initialise_csv(RAW_CSV,     RAW_FIELDS)
    initialise_csv(SUMMARY_CSV, SUMMARY_FIELDS)

    sampler = None
    if args.server_pid:
        sampler = MemorySampler(args.server_pid)
        sampler.start()
        print(f"[INFO] Memory sampler started for PID {args.server_pid}")
    else:
        print("[WARN] --server-pid not provided. Peak memory will not be tracked.")

    total     = len(prompts) * args.trials
    completed = 0
    all_rows  = [] 

    print(f"\n[INFO] {len(prompts)} prompts × {args.trials} trials = {total} runs")
    print(f"[INFO] Model      : {args.model}")
    print(f"[INFO] Threads    : {args.threads}")
    print(f"[INFO] Server URL : {args.server_url}")
    print(f"[INFO] Max tokens : {args.max_tokens}")
    print(f"[INFO] SLA        : TTFT < {SLA_TTFT_MS}ms, TPOT < {SLA_TPOT_MS}ms")
    print("=" * 60)

    for prompt in prompts:
        prompt_id = prompt["id"]
        category  = prompt["category"]
        text      = prompt["text"]

        print(f"\n[PROMPT] {prompt_id} ({category})")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        for trial in range(1, args.trials + 1):
            print(f"  Trial {trial}/{args.trials} ...", end=" ", flush=True)

            peak_mb = sampler.peak_mb if sampler else None

            try:
                result = run_inference(text, args.server_url, args.max_tokens)

                if prompt_id in MANDATORY_IDS:
                    save_mandatory_output(prompt_id, trial, result["output_text"])

                sla_str = "PASS" if result["meets_sla"] else "FAIL"

                raw_row = {
                    "prompt_id":              prompt_id,
                    "category":               category,
                    "trial":                  trial,
                    "ttft_ms":                result["ttft_ms"],
                    "tpot_ms":                result["tpot_ms"],
                    "throughput_prefill_tps": result["throughput_prefill"],
                    "throughput_decode_tps":  result["throughput_decode"],
                    "wall_time_s":            result["wall_time_s"],
                    "meets_sla":              sla_str,
                    "peak_memory_mb":         peak_mb,
                }
                append_csv_row(RAW_CSV, RAW_FIELDS, raw_row)

                all_rows.append({
                    **raw_row,
                    "meets_sla":          result["meets_sla"],
                    "throughput_prefill": result["throughput_prefill"],
                    "throughput_decode":  result["throughput_decode"],
                })

                print(
                    f"TTFT={result['ttft_ms']} ms | "
                    f"TPOT={result['tpot_ms']} ms | "
                    f"Prefill={result['throughput_prefill']} tok/s | "
                    f"Decode={result['throughput_decode']} tok/s | "
                    f"SLA={sla_str} | Wall={result['wall_time_s']} s"
                )

            except Exception as e:
                print(f"[ERROR] Trial failed: {e}")
                raw_row = {
                    "prompt_id":              prompt_id,
                    "category":               category,
                    "trial":                  trial,
                    "ttft_ms":                None,
                    "tpot_ms":                None,
                    "throughput_prefill_tps": None,
                    "throughput_decode_tps":  None,
                    "wall_time_s":            None,
                    "meets_sla":              "ERROR",
                    "peak_memory_mb":         peak_mb,
                }
                append_csv_row(RAW_CSV, RAW_FIELDS, raw_row)
                all_rows.append({
                    **raw_row,
                    "meets_sla":          None,
                    "throughput_prefill": None,
                    "throughput_decode":  None,
                })

            completed += 1

    final_peak_mb = sampler.stop() if sampler else None
    if final_peak_mb:
        print(f"\n[INFO] Peak memory (VmHWM): {final_peak_mb} MB")

    write_summary(all_rows, final_peak_mb)
    print_goodput_summary(all_rows)

    print(f"\n[DONE] {completed}/{total} runs → '{RAW_CSV}'")
    print(f"[DONE] Summary (mean ± std)  → '{SUMMARY_CSV}'")
    if any(p["id"] in MANDATORY_IDS for p in prompts):
        print(f"[DONE] Mandatory outputs     → '{MANDATORY_OUTPUT_DIR}/'")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
