import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Criar a pasta para os extras
OUT_EXTRA = os.path.join("graficos_relatorio", "5_Extras_Avancados")
os.makedirs(OUT_EXTRA, exist_ok=True)

dados = []

# Ler os ficheiros exatamente da mesma forma
for root, dirs, files in os.walk("Engine_Llama"):
    if "summary_benchmarks.csv" in files and "__MACOSX" not in root:
        filepath = os.path.join(root, "summary_benchmarks.csv")
        partes = filepath.split(os.sep)

        try:
            res_idx = next(i for i, p in enumerate(partes) if p.startswith("Resultados_"))
            modelo = partes[res_idx].replace("Resultados_", "")
            experiencia = partes[res_idx + 1]
            configuracao = partes[res_idx + 2]

            df = pd.read_csv(filepath)
            df['Model'] = modelo
            df['Experiment'] = experiencia
            df['Config'] = configuracao

            if "x86_" in configuracao or "arm_" in configuracao:
                arch, threads = configuracao.split("_")
                df['Arch'] = arch
                df['Threads'] = int(threads)
            else:
                df['Arch'] = 'x86'
                df['Threads'] = 16

            dados.append(df)
        except Exception as e:
            pass

df_all = pd.concat(dados, ignore_index=True)
df_threads = df_all[(df_all['Experiment'] == 'Threading') & (df_all['Arch'] == 'x86')]

if not df_threads.empty:
    # -------------------------------------------------------------------------
    # 1. TPOT vs Threads (Time Per Output Token)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_threads, x='Threads', y='tpot_mean_ms', hue='Model', marker='^', linewidth=2.5, markersize=8)
    plt.title("TPOT (Tempo por Token Gerado) vs Número de Threads", fontsize=14)
    plt.ylabel("TPOT Médio (ms) - Mais baixo é melhor")
    plt.xlabel("Número de Threads")
    plt.xticks([4, 8, 16, 32])
    plt.savefig(f"{OUT_EXTRA}/1_TPOT_vs_Threads.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Goodput vs Threads (% de sucesso no SLA)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_threads, x='Threads', y='goodput_pct', hue='Model', marker='D', linewidth=2.5, markersize=8)
    plt.title("Goodput (% de pedidos que cumprem o SLA) vs Threads", fontsize=14)
    plt.ylabel("Goodput (%)")
    plt.xlabel("Número de Threads")
    plt.ylim(0, 105)  # Goodput vai de 0 a 100%
    plt.xticks([4, 8, 16, 32])
    plt.savefig(f"{OUT_EXTRA}/2_Goodput_vs_Threads.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Prefill vs Decode Throughput (Comparação de Fases) para a baseline (16 cores)
    # -------------------------------------------------------------------------
    df_16t = df_threads[df_threads['Threads'] == 16].copy()
    if not df_16t.empty:
        # Preparar os dados para um gráfico de barras agrupado (Prefill vs Decode lado a lado)
        df_melted = df_16t.melt(id_vars=['Model', 'category'],
                                value_vars=['prefill_mean_tps', 'decode_mean_tps'],
                                var_name='Fase', value_name='Throughput (TPS)')
        # Limpar os nomes das fases para ficar bonito na legenda
        df_melted['Fase'] = df_melted['Fase'].map(
            {'prefill_mean_tps': 'Prefill (Ler Prompt)', 'decode_mean_tps': 'Decode (Gerar)'})

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x='Model', y='Throughput (TPS)', hue='Fase', errorbar=None)
        plt.title("Comparação de Velocidade: Processar Contexto (Prefill) vs Gerar Texto (Decode)", fontsize=14)
        plt.ylabel("Tokens / Segundo")
        plt.xlabel("Modelo")
        plt.savefig(f"{OUT_EXTRA}/3_Prefill_vs_Decode_16t.png", dpi=300, bbox_inches='tight')
        plt.close()

print(f"[SUCESSO] Mais 3 gráficos avançados criados em '{OUT_EXTRA}/'!")