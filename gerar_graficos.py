import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração visual profissional
sns.set_theme(style="whitegrid")

# Criar a estrutura de pastas para o Output
OUT_BASE = "graficos_relatorio"
dirs_out = {
    "Threading": os.path.join(OUT_BASE, "1_Threading"),
    "ArmX86": os.path.join(OUT_BASE, "2_Arm_vs_X86"),
    "Quantizacao": os.path.join(OUT_BASE, "3_Quantizacao"),
    "Geral": os.path.join(OUT_BASE, "4_Geral")
}

for d in dirs_out.values():
    os.makedirs(d, exist_ok=True)

dados = []

# 1. Percorrer a estrutura de pastas e ler os CSVs ignorando o lixo do Mac
for root, dirs, files in os.walk("Engine_Llama"):
    if "summary_benchmarks.csv" in files and "__MACOSX" not in root:
        filepath = os.path.join(root, "summary_benchmarks.csv")
        partes = filepath.split(os.sep)

        try:
            # Identificar dinamicamente onde estão as tags na framework de pastas
            res_idx = next(i for i, p in enumerate(partes) if p.startswith("Resultados_"))
            modelo = partes[res_idx].replace("Resultados_", "")
            experiencia = partes[res_idx + 1]  # Ex: Arm_X86, Threading, Quantizaçao
            configuracao = partes[res_idx + 2]  # Ex: x86_16, arm_16, Q2, Q8

            df = pd.read_csv(filepath)
            df['Model'] = modelo
            df['Experiment'] = experiencia
            df['Config'] = configuracao

            # Tentar extrair threads se aplicável
            if "x86_" in configuracao or "arm_" in configuracao:
                arch, threads = configuracao.split("_")
                df['Arch'] = arch
                df['Threads'] = int(threads)
            else:
                df['Arch'] = 'x86'  # Assumir x86 para quantização
                df['Threads'] = 16

            dados.append(df)
        except Exception as e:
            print(f"[AVISO] Ignorado {filepath}: {e}")

df_all = pd.concat(dados, ignore_index=True)
print(f"[INFO] Foram carregados dados de {len(dados)} configurações válidas.")

# =============================================================================
# 1. GRÁFICOS DE THREADING
# =============================================================================
df_threads = df_all[(df_all['Experiment'] == 'Threading') & (df_all['Arch'] == 'x86')]
if not df_threads.empty:
    # Throughput
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_threads, x='Threads', y='decode_mean_tps', hue='Model', marker='o', linewidth=2.5,
                 errorbar=None)
    plt.title("Throughput de Decode vs Threads", fontsize=14)
    plt.ylabel("Tokens / Segundo")
    plt.xticks([4, 8, 16, 32])
    plt.savefig(f"{dirs_out['Threading']}/Throughput_vs_Threads.png", dpi=300, bbox_inches='tight')
    plt.close()

    # TTFT
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_threads, x='Threads', y='ttft_mean_ms', hue='Model', marker='s', linewidth=2.5, errorbar=None)
    plt.title("TTFT vs Threads (Latência Inicial)", fontsize=14)
    plt.ylabel("Time To First Token (ms)")
    plt.xticks([4, 8, 16, 32])
    plt.savefig(f"{dirs_out['Threading']}/TTFT_vs_Threads.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 2. GRÁFICOS ARM VS X86
# =============================================================================
df_armx86 = df_all[(df_all['Experiment'] == 'Arm_X86')]
if not df_armx86.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_armx86, x='Arch', y='decode_mean_tps', hue='Model', errorbar=None)
    plt.title("Comparação de Throughput: ARM vs x86 (16 Threads)", fontsize=14)
    plt.ylabel("Tokens / Segundo")
    plt.xlabel("Arquitetura")
    plt.savefig(f"{dirs_out['ArmX86']}/Throughput_Arm_vs_x86.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_armx86, x='Arch', y='ttft_mean_ms', hue='Model', errorbar=None)
    plt.title("Comparação de TTFT: ARM vs x86 (16 Threads)", fontsize=14)
    plt.ylabel("TTFT Médio (ms)")
    plt.xlabel("Arquitetura")
    plt.savefig(f"{dirs_out['ArmX86']}/TTFT_Arm_vs_x86.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 3. GRÁFICOS DE QUANTIZAÇÃO (Q2 vs Baseline Q4 vs Q8)
# =============================================================================
df_quant = df_all[df_all['Experiment'] == 'Quantizaçao'].copy()
# Vamos buscar o baseline (x86_16 do Threading) para agir como Q4
df_baseline = df_threads[df_threads['Threads'] == 16].copy()
df_baseline['Config'] = 'Q4 (Base)'

if not df_quant.empty:
    df_q_full = pd.concat([df_quant, df_baseline])
    ordem_q = ['Q2', 'Q4 (Base)', 'Q8']  # Ordenar lógicamente

    # Peak Memory vs Quantização
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_q_full, x='Config', y='peak_memory_mb', hue='Model', order=ordem_q, errorbar=None)
    plt.title("Consumo de Memória Máxima vs Nível de Quantização", fontsize=14)
    plt.ylabel("Memória (MB)")
    plt.xlabel("Quantização")
    plt.savefig(f"{dirs_out['Quantizacao']}/Memory_vs_Quantization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Throughput vs Quantização
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_q_full, x='Config', y='decode_mean_tps', hue='Model', order=ordem_q, errorbar=None)
    plt.title("Throughput de Decode vs Nível de Quantização", fontsize=14)
    plt.ylabel("Tokens / Segundo")
    plt.xlabel("Quantização")
    plt.savefig(f"{dirs_out['Quantizacao']}/Throughput_vs_Quantization.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 4. GRÁFICOS GERAIS (Exigidos pelo Relatório)
# =============================================================================
df_base = df_threads[df_threads['Threads'] == 16]  # A baseline padrão para geral
if not df_base.empty:
    # TTFT vs Tamanho do Prompt
    plt.figure(figsize=(10, 6))
    ordem_cat = ['short', 'medium', 'long']
    sns.barplot(data=df_base, x='category', y='ttft_mean_ms', hue='Model', order=ordem_cat, errorbar=None)
    plt.title("TTFT vs Tamanho do Prompt", fontsize=14)
    plt.ylabel("Time To First Token (ms)")
    plt.xlabel("Tamanho do Contexto")
    plt.savefig(f"{dirs_out['Geral']}/TTFT_vs_Prompt_Length.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Peak Memory Overview
    plt.figure(figsize=(8, 6))
    df_mem = df_base.groupby('Model')['peak_memory_mb'].mean().reset_index()
    sns.barplot(data=df_mem, x='Model', y='peak_memory_mb', palette='viridis', hue='Model', legend=False)
    plt.title("Comparação do Consumo Máximo de RAM (16 Threads, Q4)", fontsize=14)
    plt.ylabel("Memória Máxima (MB)")
    for index, row in df_mem.iterrows():
        plt.text(index, row['peak_memory_mb'] + 50, f"{int(row['peak_memory_mb'])} MB", color='black', ha="center")
    plt.savefig(f"{dirs_out['Geral']}/Peak_Memory_Overview.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"[SUCESSO] Gráficos separados por pastas em '{OUT_BASE}/'!")