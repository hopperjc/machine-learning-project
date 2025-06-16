import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt

# 1) Carrega os resultados (sem AutoML)
df = pd.read_csv(r'Resultados_completos.csv')

# 2) Drop de linhas com qualquer NaN nas métricas de teste
df = df.dropna(subset=['mean_accuracy', 'mean_cross_entropy', 'mean_auc_ovo'])

# 3) Pivot tables “wide” (linhas=datasets, colunas=modelos)
acc_wide  = df.pivot(index='openml_id', columns='modelo', values='mean_accuracy')
ce_wide   = df.pivot(index='openml_id', columns='modelo', values='mean_cross_entropy')
auc_wide  = df.pivot(index='openml_id', columns='modelo', values='mean_auc_ovo')

# 4) Teste de Friedman
print("=== TESTE DE FRIEDMAN ===")
acc_arrays = [acc_wide[col].values for col in acc_wide.columns]
stat_acc, pval_acc = friedmanchisquare(*acc_arrays)
print(f"Accuracy  → estatística={stat_acc:.4f}, p-valor={pval_acc:.4e}")

auc_arrays = [auc_wide[col].values for col in auc_wide.columns]
stat_auc, pval_auc = friedmanchisquare(*auc_arrays)
print(f"AUC OVO   → estatística={stat_auc:.4f}, p-valor={pval_auc:.4e}")

ce_arrays  = [ce_wide[col].values for col in ce_wide.columns]
stat_ce, pval_ce = friedmanchisquare(*ce_arrays)
print(f"CrossEnt. → estatística={stat_ce:.4f}, p-valor={pval_ce:.4e}")

# 5) Se p-valor < 0.05, rodar Nemenyi para pares que diferem
if pval_acc < 0.05:
    nemenyi_acc = sp.posthoc_nemenyi_friedman(acc_wide.values)
    nemenyi_acc.index  = acc_wide.columns
    nemenyi_acc.columns = acc_wide.columns
    print("\nNemenyi (ACC) — p-valores:\n", nemenyi_acc)

if pval_auc < 0.05:
    nemenyi_auc = sp.posthoc_nemenyi_friedman(auc_wide.values)
    nemenyi_auc.index  = auc_wide.columns
    nemenyi_auc.columns = auc_wide.columns
    print("\nNemenyi (AUC) — p-valores:\n", nemenyi_auc)

if pval_ce < 0.05:
    nemenyi_ce = sp.posthoc_nemenyi_friedman(ce_wide.values)
    nemenyi_ce.index  = ce_wide.columns
    nemenyi_ce.columns = ce_wide.columns
    print("\nNemenyi (CE) — p-valores:\n", nemenyi_ce)

# 6) Plot de Rank Médio para cada métrica
def plot_ranking(df_wide, ascending, titulo):
    ranks = df_wide.rank(axis=1, method='average', ascending=ascending)
    rank_medio = ranks.mean(axis=0).sort_values()
    plt.figure(figsize=(6,4))
    plt.barh(rank_medio.index, rank_medio.values)
    plt.xlabel('Rank Médio')
    plt.title(titulo)
    plt.gca().invert_xaxis()  # rank 1 (melhor) no topo
    plt.tight_layout()
    plt.show()

plot_ranking(acc_wide, ascending=False, titulo='Rank Médio por ACC (sem AutoML)')
plot_ranking(auc_wide, ascending=False, titulo='Rank Médio por AUC OVO (sem AutoML)')
plot_ranking(ce_wide,  ascending=True,  titulo='Rank Médio por Cross-Entropy (sem AutoML)')