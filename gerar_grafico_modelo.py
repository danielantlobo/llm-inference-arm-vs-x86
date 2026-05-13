import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração visual
sns.set_theme(style="whitegrid")

# =============================================================================
# 1. PARÂMETROS DO MODELO MATEMÁTICO (Ajusta consoante os teus dados exatos)
# =============================================================================

# Tamanho físico de cada modelo em GB (GigaBytes) para as versões Q4_K_M
MODEL_SIZES_GB = {
    "Meta_Lama": 4.6,  # Llama 3.1 8B Q4 costuma ter ~4.92 GB
    "Qwen": 1.10,  # Qwen 1.5B Q4 costuma ter ~1.12 GB
    "Tiny_Lama": 0.638,  # TinyLlama 1.1B Q4 costuma ter ~0.73 GB
    "Smol": 0.25  # SmolLM 360M Q4 (caso o tenhas usado nalgum lado)
}

# Largura de banda teórica da memória RAM (Memory Bandwidth) em GB/s
# Um nó x86 típico do Deucalion (EPYC/Xeon dual socket) atinge frequentemente ~200 GB/s
MEMORY_BANDWIDTH_GB_S = 204.8

# =============================================================================

OUT_MODEL = os.path.join("graficos_relatorio", "6_Performance_Model")
os.makedirs(OUT_MODEL, exist_ok=True)

dados = []

# 2. Ler os dados (vamos usar apenas a Baseline de 16 threads no x86 em Q4)
for root, dirs, files in os.walk("Engine_Llama"):
    if "summary_benchmarks.csv" in files and "__MACOSX" not in root:
        filepath = os.path.join(root, "summary_benchmarks.csv")
        partes = filepath.split(os.sep)

        try:
            res_idx = next(i for i, p in enumerate(partes) if p.startswith("Resultados_"))
            modelo = partes[res_idx].replace("Resultados_", "")
            experiencia = partes[res_idx + 1]
            configuracao = partes[res_idx + 2]

            # Só queremos a baseline x86 com 16 threads (experiência base)
            if experiencia == "Threading" and configuracao == "x86_16":
                df = pd.read_csv(filepath)
                df['Model'] = modelo
                dados.append(df)
        except Exception as e:
            pass

if not dados:
    print("[ERRO] Não foram encontrados dados para a baseline x86_16.")
    exit()

df_baseline = pd.concat(dados, ignore_index=True)

# Agrupar pelo modelo para obter o TPOT real observado (média de todos os prompts)
df_grouped = df_baseline.groupby('Model')['tpot_mean_ms'].mean().reset_index()
df_grouped.rename(columns={'tpot_mean_ms': 'Observed TPOT (ms)'}, inplace=True)


# 3. Calcular o TPOT Previsto (Predicted TPOT)
def calcular_predicted_tpot(modelo_nome):
    # Encontrar o tamanho correspondente no dicionário
    size_gb = 1.0  # Default fallback
    for key, val in MODEL_SIZES_GB.items():
        if key in modelo_nome:
            size_gb = val
            break

    # Fórmula: TPOT (ms) = (Size_GB / Bandwidth_GB_s) * 1000
    return (size_gb / MEMORY_BANDWIDTH_GB_S) * 1000


df_grouped['Predicted TPOT (ms)'] = df_grouped['Model'].apply(calcular_predicted_tpot)

# 4. Preparar os dados para o gráfico (Melt)
df_plot = df_grouped.melt(id_vars=['Model'],
                          value_vars=['Observed TPOT (ms)', 'Predicted TPOT (ms)'],
                          var_name='Tipo', value_name='TPOT (ms)')

# 5. Gerar o Gráfico de Barras Agrupadas
plt.figure(figsize=(10, 6))
grafico = sns.barplot(data=df_plot, x='Model', y='TPOT (ms)', hue='Tipo', palette=['#ff7f0e', '#1f77b4'])

plt.title("Performance Model: Predicted vs. Observed TPOT (16 Threads)", fontsize=14, fontweight='bold')
plt.ylabel("Time Per Output Token - TPOT (ms)", fontsize=12)
plt.xlabel("Modelo Analisado", fontsize=12)

# Adicionar os valores em cima de cada barra para clareza
for p in grafico.patches:
    height = p.get_height()
    if height > 0:
        grafico.annotate(f'{height:.1f} ms',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom',
                         xytext=(0, 5), textcoords='offset points',
                         fontsize=10)

plt.legend(title="Legenda")
plt.tight_layout()

# Guardar o gráfico
caminho_grafico = os.path.join(OUT_MODEL, "Predicted_vs_Observed_TPOT.png")
plt.savefig(caminho_grafico, dpi=300)
plt.close()

print(f"\n[SUCESSO] Gráfico do Modelo de Performance gerado em: {caminho_grafico}")
print("\n[INFO] Tabela Resumo dos Dados:")
print(df_grouped.to_string(index=False))