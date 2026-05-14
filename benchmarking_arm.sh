#!/bin/bash
#SBATCH --job-name=benchmarking-arm
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

module load GCCcore/13.3.0 Python/3.12.3
module load GCC/13.3.0

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR="/projects/F202500010HPCVLABUMINHO/uminhocp022/aded2"
BASELINE_M="$BASE_DIR/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
SERVER_BIN="$BASE_DIR/llama.cpp/build_arm/bin/llama-server"
THREADS=$SLURM_CPUS_PER_TASK
PORT=8080
SERVER_URL="http://localhost:$PORT"

# ─────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────

mkdir -p $BASE_DIR/results/arm2/metrics
mkdir -p $BASE_DIR/results/arm2/outputs

# ─────────────────────────────────────────────
# RESOURCE MONITORING (background)
# ─────────────────────────────────────────────

pidstat -u -r 1 > $BASE_DIR/results/arm2/metrics/recursos_sistema.log &
PIDSTAT_PID=$!

# ─────────────────────────────────────────────
# START LLAMA-SERVER (model loads ONCE)
# ─────────────────────────────────────────────

echo "[SLURM] Starting llama-server on port $PORT with $THREADS threads..."

$SERVER_BIN \
    --model "$BASELINE_M" \
    --threads $THREADS \
    --port $PORT \
    --ctx-size 4096 \
    --log-disable \
    &

SERVER_PID=$!
echo "[SLURM] llama-server PID: $SERVER_PID"

# ─────────────────────────────────────────────
# WAIT FOR SERVER TO BE READY
# ─────────────────────────────────────────────

echo "[SLURM] Waiting for server to load model..."
MAX_WAIT=180
WAITED=0

until curl -s "$SERVER_URL/health" | grep -q '"status":"ok"'; do
    sleep 3
    WAITED=$((WAITED + 3))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[ERROR] Server did not start within ${MAX_WAIT}s. Aborting."
        kill $SERVER_PID 2>/dev/null
        kill $PIDSTAT_PID 2>/dev/null
        exit 1
    fi
    echo "[SLURM] Still waiting... (${WAITED}s)"
done

echo "[SLURM] Server ready after ${WAITED}s."

# ─────────────────────────────────────────────
# WARMUP (1 request — not logged)
# ─────────────────────────────────────────────

echo "[SLURM] Running warmup request..."
curl -s "$SERVER_URL/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "n_predict": 16, "temperature": 0.7}' \
    > /dev/null
echo "[SLURM] Warmup done."

# ─────────────────────────────────────────────
# RUN BENCHMARK
# SERVER_PID passed so Python can track VmHWM
# ─────────────────────────────────────────────

echo "[SLURM] Starting benchmark (server PID: $SERVER_PID)..."

python3 -u $BASE_DIR/benchmarker.py \
    --threads $THREADS \
    --model "$BASELINE_M" \
    --server-url "$SERVER_URL" \
    --server-pid $SERVER_PID

echo "[SLURM] Benchmark complete."

# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────

kill $SERVER_PID 2>/dev/null
kill $PIDSTAT_PID 2>/dev/null

# Record SLURM job efficiency (sacct is more reliable than seff across partitions)
sacct -j $SLURM_JOB_ID \
    --format=JobID,Elapsed,CPUTime,MaxRSS,AveRSS,ExitCode \
    >> $BASE_DIR/results/arm2/metrics/slurm_efficiency.txt 2>/dev/null || \
    echo "sacct unavailable on this node" \
    >> $BASE_DIR/results/arm2/metrics/slurm_efficiency.txt
