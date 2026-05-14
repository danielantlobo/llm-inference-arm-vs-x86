#!/bin/bash
#SBATCH --job-name=compiling-arm
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00

module load GCC/13.3.0 CMake/3.29.3

cd llama.cpp

rm -rf build_arm
mkdir -p build_arm
cd build_arm

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_LTO=ON

make -j $SLURM_CPUS_PER_TASK

echo "=== Building llama-server ==="
make -j $SLURM_CPUS_PER_TASK llama-server

echo ""
echo "=== Build complete. Verifying binary ==="
if [ -f "bin/llama-server" ]; then
    echo "[OK] llama-server compiled successfully!"
    bin/llama-server --version
else
    echo "[ERROR] llama-server binary not found!"
fi
