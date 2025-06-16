import pandas as pd
import glob

# Carregue o DataFrame com os resultados dos seus modelos base
df_base = pd.read_csv("resultados_cc18_sem_automl_paralelo.csv") # Use o nome correto do seu arquivo

# Carregue todos os arquivos de resultado do AutoGluon (se você rodou em lotes)
autogluon_files = glob.glob("resultados_autogluon*.csv")
if not autogluon_files:
    print("Nenhum arquivo de resultado do AutoGluon encontrado. Verifique os nomes dos arquivos.")
else:
    df_autogluon_list = [pd.read_csv(f) for f in autogluon_files]
    df_autogluon = pd.concat(df_autogluon_list, ignore_index=True)

    # Concatena todos em um único DataFrame final
    df_completo = pd.concat([df_base, df_autogluon], ignore_index=True)

    # Salva o arquivo mestre com todos os resultados
    df_completo.to_csv("resultados_completos_com_automl.csv", index=False)
    df_autogluon.to_csv("resultados_autogluon.csv", index=False)

    print("Arquivo 'resultados_completos_com_automl.csv' criado com sucesso!")
    print(df_completo.head())
