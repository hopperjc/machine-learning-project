import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare, t
from scipy import stats
import scikit_posthocs as sp
import autorank

# --- Carregamento e Preparação dos Dados ---

# 1) Carrega os resultados completos
# Certifique-se que o nome do arquivo CSV está correto
try:
    df = pd.read_csv(r'Resultados_completos.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'resultados_completos_com_automl.csv' não encontrado.")
    print("Certifique-se de que o arquivo com os resultados completos está na mesma pasta que este script.")
    exit()

# --- GERAÇÃO DAS TABELAS ---

# Tabela 1: Média, Desvio Padrão e Intervalo de Confiança
print("Gerando Tabelas de Métricas...")


def get_full_stats(data):
    """Calcula média, std, e IC 95% para uma série de dados."""
    n = len(data)
    if n < 2:
        return pd.Series({
            'mean': np.mean(data), 'std': np.nan,
            'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
            'margin_of_error': np.nan
        })

    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 para desvio padrão amostral
    sem = stats.sem(data)

    # Margem de erro para o IC de 95%
    margin_of_error = sem * t.ppf((1 + 0.95) / 2., n - 1)

    return pd.Series({
        'mean': mean, 'std': std,
        'ci_95_lower': mean - margin_of_error,
        'ci_95_upper': mean + margin_of_error,
        'margin_of_error': margin_of_error
    })


metrics_to_agg = ['mean_accuracy', 'mean_auc_ovo', 'mean_cross_entropy', 'total_time_sec']
stats_list = []

for model_name, model_group in df.groupby('modelo'):
    model_stats = model_group[metrics_to_agg].apply(get_full_stats).T
    model_stats['modelo'] = model_name
    stats_list.append(model_stats)

# Tabela 1a: Todos os valores numéricos detalhados
full_stats_df = pd.concat(stats_list).reset_index().rename(columns={'index': 'metrica'})
full_stats_df = full_stats_df[['modelo', 'metrica', 'mean', 'std', 'ci_95_lower', 'ci_95_upper', 'margin_of_error']]
full_stats_df.to_csv('tabela_1a_metricas_detalhadas.csv', index=False, float_format='%.4f')
print("Salvo em 'tabela_1a_metricas_detalhadas.csv'")

# Tabela 1b: Tabela formatada para o relatório (média ± margem de erro)
full_stats_df['valor_formatado'] = full_stats_df.apply(
    lambda row: f"{row['mean']:.4f} ± {row['margin_of_error']:.4f}", axis=1
)
formatted_table = full_stats_df.pivot(index='modelo', columns='metrica', values='valor_formatado')
# Reordena as colunas para a ordem desejada
formatted_table = formatted_table[['mean_accuracy', 'mean_auc_ovo', 'mean_cross_entropy', 'total_time_sec']]
formatted_table.to_csv('tabela_1b_metricas_formatadas.csv')
print("Salvo em 'tabela_1b_metricas_formatadas.csv'\n")

# Preparação para os testes estatísticos
# 3) Pivot tables “wide” (linhas=datasets, colunas=modelos)
acc_wide = df.pivot(index='openml_id', columns='modelo', values='mean_accuracy')
ce_wide = df.pivot(index='openml_id', columns='modelo', values='mean_cross_entropy')
auc_wide = df.pivot(index='openml_id', columns='modelo', values='mean_auc_ovo')


print(f"Executando testes em {len(acc_wide)} de {len(acc_wide)} datasets onde todos os modelos tiveram sucesso.\n")

# Tabela 2: Ranking Médio para cada Métrica
print("Gerando Tabela 2: Ranking Médio...")
ranks_acc = acc_wide.rank(axis=1, method='average', ascending=False)  # Para ACC, maior é melhor
ranks_auc = auc_wide.rank(axis=1, method='average', ascending=False)  # Para AUC, maior é melhor
ranks_ce = ce_wide.rank(axis=1, method='average', ascending=True)  # Para Cross-Entropy, menor é melhor

mean_ranks_df = pd.DataFrame({
    'ACC_Rank_Medio': ranks_acc.mean(axis=0),
    'AUC_OVO_Rank_Medio': ranks_auc.mean(axis=0),
    'CE_Rank_Medio': ranks_ce.mean(axis=0)
}).sort_values(by='AUC_OVO_Rank_Medio').round(4)  # Ordena pelo rank do AUC
mean_ranks_df.to_csv('tabela_2_ranks_medios.csv')
print("Salvo em 'tabela_2_ranks_medios.csv'\n")

# Tabela 3: Teste de Friedman
print("Gerando Tabela 3: Teste de Friedman...")
stat_acc, pval_acc = friedmanchisquare(*[acc_wide[col].values for col in acc_wide.columns])
stat_auc, pval_auc = friedmanchisquare(*[auc_wide[col].values for col in auc_wide.columns])
# Para Cross-Entropy, ranks menores são melhores. Invertemos os valores para o teste.
stat_ce, pval_ce = friedmanchisquare(*[-ce_wide[col].values for col in ce_wide.columns])

friedman_results = {
    'Métrica': ['Accuracy', 'AUC OVO', 'Cross-Entropy'],
    'Estatística': [stat_acc, stat_auc, stat_ce],
    'p-valor': [pval_acc, pval_auc, pval_ce]
}
friedman_df = pd.DataFrame(friedman_results)
friedman_df.to_csv('tabela_3_teste_friedman.csv', index=False)
print("Salvo em 'tabela_3_teste_friedman.csv'\n")

# Tabela 4: Teste de Nemenyi (se necessário)
print("Gerando Tabela 4: Teste Post-hoc de Nemenyi (se aplicável)...")
# Se p-valor de Friedman < 0.05, rodar Nemenyi para pares que diferem
if pval_auc < 0.05:
    nemenyi_auc = sp.posthoc_nemenyi_friedman(auc_wide.values)
    nemenyi_auc.index = auc_wide.columns
    nemenyi_auc.columns = auc_wide.columns
    nemenyi_auc.to_csv('tabela_4_nemenyi_auc.csv')
    print("Teste de Nemenyi para AUC salvo em 'tabela_4_nemenyi_auc.csv'")

if pval_ce < 0.05:
    nemenyi_ce = sp.posthoc_nemenyi_friedman(ce_wide.values)
    nemenyi_ce.index = ce_wide.columns
    nemenyi_ce.columns = ce_wide.columns
    nemenyi_ce.to_csv('tabela_4_nemenyi_ce.csv')
    print("Teste de Nemenyi para Cross-Entropy salvo em 'tabela_4_nemenyi_ce.csv'")

if pval_acc < 0.05:
    nemenyi_acc = sp.posthoc_nemenyi_friedman(acc_wide.values)
    nemenyi_acc.index  = acc_wide.columns
    nemenyi_acc.columns = acc_wide.columns
    nemenyi_acc.to_csv('tabela_4_nemenyi_acc.csv')
    print("Teste de Nemenyi para ACC salvo em 'tabela_4_nemenyi_acc.csv'")

if pval_acc >= 0.05:
    print("Não foi gerada tabela de Nemenyi para Acurácia pois o p-valor de Friedman não foi significativo.")

print("\n=== ANÁLISE COM AUTORANK E DIAGRAMA DE DIFERENÇA CRÍTICA ===")

# --- Análise para AUC OVO ---
print("\n--- Resultados para AUC OVO ---")
# Gera o resultado estatístico completo
# alpha=0.05 é o nosso nível de significância
report_auc = autorank.autorank(auc_wide, alpha=0.05, verbose=False)
print(report_auc)

# Gera, mostra e salva o Diagrama de Diferença Crítica
autorank.plot_stats(report_auc)
plt.title("Diagrama de Diferença Crítica para AUC OVO")
plt.savefig("cd_diagram_auc.png", dpi=300) # Salva a imagem em alta qualidade
plt.show()


# --- Análise para Cross-Entropy ---
print("\n--- Resultados para Cross-Entropy ---")
# Para CE, ranks menores são melhores, então invertemos o sinal dos dados
# pois autorank assume que valores maiores são melhores.
report_ce = autorank.autorank(-ce_wide, alpha=0.05, verbose=False)
print(report_ce)

autorank.plot_stats(report_ce)
plt.title("Diagrama de Diferença Crítica para Cross-Entropy")
plt.savefig("cd_diagram_ce.png", dpi=300)
plt.show()

# --- Análise para Acurácia ---
# (Pode não ser muito interessante, já que o teste de Friedman não foi significativo)
print("\n--- Resultados para Acurácia ---")
report_acc = autorank.autorank(acc_wide, alpha=0.05, verbose=False)
print(report_acc)

autorank.plot_stats(report_acc)
plt.title("Diagrama de Diferença Crítica para Acurácia")
plt.savefig("cd_diagram_acc.png", dpi=300)
plt.show()