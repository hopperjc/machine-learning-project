import time
import numpy as np
import pandas as pd

import openml
from joblib import Parallel, delayed
import traceback

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from scipy.stats import loguniform

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from mlp import Standalone_RealMLP_TD_S_Classifier

def carregar_base_openml(openml_id, test_size=0.3, seed=42):
    """
    Baixa o dataset do OpenML, separa em treino/teste e retorna X_train, X_test e y_train, y_test.
    """
    dataset = openml.datasets.get_dataset(openml_id, download_all_files=False)
    X_df, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    # Se y vier como Series, converte para array
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=seed,
        stratify=(y if len(np.unique(y)) > 1 else None)
    )
    return X_train_df, X_test_df, y_train, y_test


def tunar_com_cv_melhorado(modelo_nome, X_train_df, y_train_enc, seed=42):
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # O CatBoost e o XGBoost com enable_categorical=True as detectam pelo dtype.
    categorical_cols = X_train_df.select_dtypes(include=['category']).columns.tolist()

    if modelo_nome == 'lightgbm':
        clf = LGBMClassifier(random_state=seed, force_col_wise=True, verbose=-1)
        param_dist = {
            'n_estimators': [100, 200, 500],
            'learning_rate': loguniform(0.01, 0.2),
            'num_leaves': [20, 31, 40, 50],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }
        fit_params = {'categorical_feature': categorical_cols}

    elif modelo_nome == 'xgboost':
        clf = XGBClassifier(eval_metric='mlogloss', random_state=seed,
                            tree_method='hist', enable_categorical=True)
        param_dist = {
            'n_estimators': [100, 200, 500],
            'learning_rate': loguniform(0.01, 0.2),
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }
        fit_params = {}

    elif modelo_nome == 'catboost':
        cat_features_names = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

        clf = CatBoostClassifier(verbose=False, random_seed=seed, allow_writing_files=False,
                                 cat_features=cat_features_names, thread_count=1)
        param_dist = {
            'iterations': [100, 200, 500],
            'learning_rate': loguniform(0.01, 0.2),
            'depth': [4, 6, 8],
            'l2_leaf_reg': loguniform(1, 10),
        }
        fit_params = {}

    # Usando RandomizedSearchCV para uma busca mais ampla
    rand_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=15,
        cv=cv3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=seed
    )
    rand_search.fit(X_train_df, y_train_enc, **fit_params)

    info_cv = {
        'best_params': rand_search.best_params_,
        'best_score_cv': float(rand_search.best_score_)
    }
    return rand_search.best_estimator_, info_cv


def avalia_modelo_com_cv_corrigido(openml_id, modelo_nome, seed=42):
    """
    Executa o fluxo completo de avaliação para um modelo em um dataset,
    garantindo que o pré-processamento seja específico para cada modelo.
    """
    # --- Passo 1: Carregamento e Preparação dos Dados ---
    # Carrega os dados e garante que o alvo 'y' seja um array 1D.
    X_train_df, X_test_df, y_train_orig, y_test_orig = carregar_base_openml(openml_id, 0.3, seed)

    # Codifica o alvo (y) para formato numérico (0, 1, 2...).
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_orig)
    y_test_enc = le.transform(y_test_orig)

    # Converte colunas de texto/objeto para o tipo 'category' do Pandas.
    # Essencial para que CatBoost e XGBoost usem seu tratamento nativo de categorias.
    for col in X_train_df.select_dtypes(include=['object', 'string']).columns:
        if col in X_train_df.columns:
            X_train_df[col] = X_train_df[col].astype('category')
            X_test_df[col] = X_test_df[col].astype('category')

    cat_cols = X_train_df.select_dtypes(include=['category', 'object']).columns
    if len(cat_cols) > 0:
        # Converte para 'object' (string) e preenche NaNs de uma só vez.
        # Isso evita todos os warnings de dtype incompatível do Pandas.
        for col in cat_cols:
            X_train_df.loc[:, col] = X_train_df[col].astype(str).fillna("__MISSING__")
            X_test_df.loc[:, col] = X_test_df[col].astype(str).fillna("__MISSING__")

    # --- Passo 2: Tratamento Comum de Dados Faltantes ---
    # Imputa valores numéricos faltantes usando a mediana dos dados de TREINO.
    # Isso evita data leakage do conjunto de teste.
    num_cols = X_train_df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0 and X_train_df[num_cols].isnull().sum().sum() > 0:
        imp = SimpleImputer(strategy='median')
        X_train_df[num_cols] = imp.fit_transform(X_train_df[num_cols])
        X_test_df[num_cols] = imp.transform(X_test_df[num_cols])

    # --- Passo 3: Treinamento e Avaliação Específicos por Modelo ---
    best_params_info = {}
    t0_total = time.time()  # Inicia a contagem de tempo total

    modelo_lower = modelo_nome.lower()

    if modelo_lower == 'realmlp':
        # O RealMLP recebe o DataFrame e aplica seu pipeline de pré-processamento internamente.
        clf_final = Standalone_RealMLP_TD_S_Classifier(device='cpu')
        clf_final.fit(X_train_df, y_train_enc)

    elif modelo_lower in ['lightgbm', 'xgboost', 'catboost']:
        # Os GBDTs recebem o DataFrame para a busca de hiperparâmetros e treino.
        clf_final, info_cv = tunar_com_cv_melhorado(modelo_lower, X_train_df, y_train_enc, seed)
        best_params_info.update(info_cv)
    else:
        raise ValueError(f"Modelo '{modelo_nome}' não suportado.")

    # --- Passo 4: Predição e Métricas Finais no Conjunto de Teste ---
    # O método .predict/.predict_proba de cada modelo já sabe como lidar com os dados.
    # No caso do RealMLP, ele aplicará a transformação internamente.
    y_proba = clf_final.predict_proba(X_test_df)
    y_pred = clf_final.predict(X_test_df)

    # O tempo total é medido após tune+train+predict, conforme as diretrizes.
    total_time = time.time() - t0_total

    y_true = y_test_enc
    n_classes = len(np.unique(y_true))

    # Lógica para cálculo do AUC em casos binários ou multiclasse
    if n_classes == 2:
        # Usa a probabilidade da classe positiva (coluna 1)
        auc_ovo = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc_ovo = roc_auc_score(y_true, y_proba, multi_class='ovo')

    acc = accuracy_score(y_true, y_pred)
    ce = log_loss(y_true, y_proba)

    resultados = {
        'openml_id': openml_id,
        'modelo': modelo_lower,
        'total_time_sec': float(total_time),
        'mean_auc_ovo': float(auc_ovo),
        'mean_accuracy': float(acc),
        'mean_cross_entropy': float(ce),
        **best_params_info
    }
    return resultados


if __name__ == '__main__':

    # Lista dos 30 datasets do CC18
    cc18_ids = [
            11, 15, 18, 23, 29, 31, 37, 50, 54, 188,
            307, 458, 469, 1049, 1050, 1063, 1068, 1462, 1464, 1468,
            1480, 1494, 1501, 1510, 6332, 23381, 40966, 40975, 40982, 40994
        ]
    modelos = ['realmlp', 'lightgbm', 'xgboost', 'catboost']

    # Este loop sequencial garante que todos os datasets sejam baixados antes do paralelismo.
    print("--- Aquecendo o cache: Baixando todos os datasets necessários ---")
    for oid in cc18_ids:
        try:
            openml.datasets.get_dataset(oid, download_data=True)
            print(f"Dataset {oid} OK.")
        except Exception as e:
            print(f"!!! Falha ao baixar o dataset {oid}: {e} !!!")
    print("--- Aquecimento de cache concluído ---\n")

    # 1. Criar uma lista de todas as tarefas a serem executadas
    # Cada tarefa é uma chamada à sua função de avaliação com os argumentos definidos.
    print("Preparando tarefas para execução paralela...")
    tasks = [
        delayed(avalia_modelo_com_cv_corrigido)(oid, m)
        for oid in cc18_ids
        for m in modelos
    ]

    # 2. Executar as tarefas em paralelo
    print(f"Iniciando a execução paralela de {len(tasks)} tarefas...")


    def run_safely(oid, m):
        """Wrapper para chamar a função de avaliação e capturar exceções."""
        print(f"--- Processando: {m} no dataset {oid} ---")
        try:
            return avalia_modelo_com_cv_corrigido(openml_id=oid, modelo_nome=m, seed=42)
        except Exception as e:
            print(f"!!! ERRO em {m} @ {oid}: {e} !!!")
            traceback.print_exc()  # Imprime o traceback completo do erro
            return None  # Retorna None se a tarefa falhar


    # Recria a lista de tarefas usando o wrapper seguro
    tasks_safe = [delayed(run_safely)(oid, m) for oid in cc18_ids for m in modelos]

    # Executa a versão segura em paralelo
    results_with_none = Parallel(n_jobs=-2, verbose=10)(tasks_safe)

    # 3. Filtra os resultados de tarefas que falharam
    todos_resultados = [res for res in results_with_none if res is not None]

    # 4. Monta DataFrame final e grava CSV
    if todos_resultados:
        df_all = pd.DataFrame(todos_resultados)
        df_all.to_csv('resultados_cc18_sem_automl_paralelo.csv', index=False)
        print("\n--- Tabela final (sem AutoML) ---")
        print(df_all.head(10))
    else:
        print("\nNenhuma tarefa foi concluída com sucesso.")
