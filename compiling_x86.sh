#!/bin/bash
#SBATCH --job-name=compiling-x86
#SBATCH --partition=normal-x86
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00

# 1. Carregar módulos para x86 (verificar se são os mesmos nomes no cluster)
module load GCC/13.3.0 CMake/3.29.3

# Entrar na diretoria (assume-se que o script está na raiz do projeto)
cd llama.cpp

# 2. Criar diretoria específica para x86 para evitar conflitos com o binário ARM
rm -rf build_x86
mkdir -p build_x86
cd build_x86

# 3. Configuração CMake otimizada para x86
# -DGGML_NATIVE=ON e -DLLAMA_NATIVE=ON ativam AVX/AVX2/AVX512 automaticamente no x86
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_LTO=ON

# 4. Compilação
echo "=== Building llama-server for x86 ==="
make -j $SLURM_CPUS_PER_TASK llama-server

echo ""
echo "=== Build complete. Verifying binary ==="
# O caminho pode variar dependendo da versão do llama.cpp (bin/ ou apenas o nome do executável)
if [ -f "bin/llama-server" ]; then
    echo "[OK] llama-server compiled successfully for x86!"
    bin/llama-server --version
elif [ -f "llama-server" ]; then
    echo "[OK] llama-server compiled successfully for x86!"
    ./llama-server --version
else
    echo "[ERROR] llama-server binary not found!"
    ls -R # Ajuda a depurar onde o binário foi parar caso falhe
fi
