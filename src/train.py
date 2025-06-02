import time
import numpy  as np
import pandas as pd

import openml

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import roc_auc_score, accuracy_score, log_loss

from lightgbm  import LGBMClassifier
from xgboost   import XGBClassifier
from catboost  import CatBoostClassifier

from mlp               import Standalone_RealMLP_TD_S_Classifier
from preprocessing     import get_realmlp_td_s_pipeline


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
        y = y.values.reshape(-1,)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=seed,
        stratify=(y if len(np.unique(y)) > 1 else None)
    )
    return X_train_df, X_test_df, y_train, y_test


def tunar_com_cv(modelo_nome, X_train_np, y_train_enc, seed=42):
    """
    Recebe o nome do modelo ('lightgbm', 'xgboost' ou 'catboost'), faz GridSearchCV em 3 folds
    para encontrar os melhores hiperparâmetros e retorna o estimador ajustado + informações sobre o CV.
    """
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    if modelo_nome == 'lightgbm':
        clf = LGBMClassifier(random_state=seed)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }

    elif modelo_nome == 'xgboost':
        clf = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=seed
        )
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }

    elif modelo_nome == 'catboost':
        # Desliga a criação de catboost_info para evitar erro de permissão
        clf = CatBoostClassifier(
            verbose=False,
            random_seed=seed,
            allow_writing_files=False
        )
        param_grid = {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1]
        }

    else:
        return None, {}

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        error_score='raise'
    )
    grid.fit(X_train_np, y_train_enc)

    info_cv = {
        'best_params':    grid.best_params_,
        'best_score_cv':  float(grid.best_score_)
    }
    return grid.best_estimator_, info_cv


def avalia_modelo_com_cv(openml_id, modelo_nome, seed=42):
    """
    Carrega dados, faz pré‐processamento, ajusta o modelo especificado com CV interno (se aplicável),
    calcula métricas de teste e retorna um dict com resultados completos.
    """
    # 1) Carrega dados originais (y em string, ex.: “good”/“bad”)
    X_train_df, X_test_df, y_train_orig, y_test_orig = carregar_base_openml(openml_id, 0.3, seed)

    # 2) LabelEncode para 0,1,... (evita erro “Invalid classes … got ['bad' 'good']”)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_orig)
    y_test_enc  = le.transform(y_test_orig)

    # 3) Imputação das colunas numéricas (se houver)
    num_cols = X_train_df.select_dtypes(exclude=['object','category','string']).columns
    if len(num_cols) > 0:
        imp = SimpleImputer(strategy='median')
        X_train_df[num_cols] = imp.fit_transform(X_train_df[num_cols])
        X_test_df[num_cols]  = imp.transform(X_test_df[num_cols])

    # 4) Pipeline RealMLP (CustomOneHot + RobustScale) → output numpy array
    pipeline = get_realmlp_td_s_pipeline()
    X_train_np = pipeline.fit_transform(X_train_df.values)
    X_test_np  = pipeline.transform( X_test_df.values)

    # 5) Define modelo e faz CV interno conforme tipo
    best_params_info = {}
    clf_final        = None

    modelo_lower = modelo_nome.lower()
    if modelo_lower == 'realmlp':
        # MLP customizado (não faz busca de hiperparâmetros aqui)
        clf_tmp = Standalone_RealMLP_TD_S_Classifier(device='cpu')
        cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        # Mede acurácia no CV de treino
        scores_acc = cross_val_score(
            clf_tmp, X_train_np, y_train_enc,
            cv=cv3, scoring='accuracy', n_jobs=1
        )
        best_params_info['cv_accuracy_mean'] = float(np.mean(scores_acc))
        best_params_info['cv_accuracy_std']  = float(np.std(scores_acc))

        # Ajusta modelo no dataset de treino completo
        clf_final = Standalone_RealMLP_TD_S_Classifier(device='cpu')
        t0 = time.time()
        clf_final.fit(X_train_np, y_train_enc, X_val=None, y_val=None)
        treino_time = time.time() - t0

        # Faz inferência no conjunto de teste
        t1 = time.time()
        y_pred_enc  = clf_final.predict(X_test_np)
        y_proba_enc = clf_final.predict_proba(X_test_np)
        infer_time  = time.time() - t1

        y_true = y_test_enc

    elif modelo_lower in ['lightgbm', 'xgboost', 'catboost']:
        # Busca de hiperparâmetros via CV interno
        estimator_tunado, info_cv = tunar_com_cv(modelo_lower, X_train_np, y_train_enc, seed=seed)
        best_params_info.update(info_cv)

        # Ajusta o estimador tunado no conjunto de treino
        t0 = time.time()
        estimator_tunado.fit(X_train_np, y_train_enc)
        treino_time = time.time() - t0

        # Faz inferência no conjunto de teste
        t1 = time.time()
        y_proba_enc = estimator_tunado.predict_proba(X_test_np)
        y_pred_enc  = estimator_tunado.predict(X_test_np)
        infer_time  = time.time() - t1

        clf_final = estimator_tunado
        y_true    = y_test_enc

    else:
        raise ValueError(
            f"Modelo '{modelo_nome}' não suportado. "
            "Use apenas: realmlp, lightgbm, xgboost ou catboost."
        )

    # 6) Métricas no teste final
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        auc_ovo = roc_auc_score(y_true, y_proba_enc[:, 1])
    else:
        auc_ovo = roc_auc_score(y_true, y_proba_enc, multi_class='ovo')

    acc = accuracy_score(y_true, y_pred_enc)
    ce  = log_loss(y_true, y_proba_enc)

    resultados = {
        'openml_id':          openml_id,
        'modelo':             modelo_lower,
        'n_train':            len(y_train_enc),
        'n_test':             len(y_test_enc),
        'n_classes':          n_classes,
        'treino_time_sec':    float(treino_time),
        'infer_time_sec':     float(infer_time),
        'mean_auc_ovo':       float(auc_ovo),
        'mean_accuracy':      float(acc),
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

    todos_resultados = []
    for oid in cc18_ids:
        for m in modelos:
            print(f"--- Avaliando {m} no dataset {oid} com CV interno ---")
            try:
                res = avalia_modelo_com_cv(oid, modelo_nome=m, seed=42)
                todos_resultados.append(res)
            except Exception as e:
                print(f"Erro em {m} @ {oid}: {e}")
                continue

    # Monta DataFrame final e grava CSV
    df_all = pd.DataFrame(todos_resultados)
    df_all.to_csv('resultados_cc18_sem_automl_cv.csv', index=False)

    print("\nTabela final (sem AutoML):")
    print(df_all.head(10))