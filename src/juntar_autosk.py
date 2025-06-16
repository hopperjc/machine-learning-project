import pandas as pd
import glob

# Carregue o DataFrame com os resultados dos seus modelos base

# Carregue todos os arquivos de resultado do AutoGluon (se você rodou em lotes)
autosklearn_files = glob.glob("results_autosklearn/resultados_autosklearn*.csv")

df_autosklearn_list = [pd.read_csv(f) for f in autosklearn_files]
df_autosklearn = pd.concat(df_autosklearn_list, ignore_index=True)


# Salva o arquivo mestre com todos os resultados
df_autosklearn.to_csv("resultados_autosklearn.csv", index=False)
