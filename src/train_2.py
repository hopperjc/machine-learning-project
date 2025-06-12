import time
import numpy as np
import pandas as pd
import warnings
import openml

# Modelos
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Modelos e pré-processamento customizados (seus arquivos mlp.py e preprocessing.py)
from mlp import Standalone_RealMLP_TD_S_Classifier

# Utilitários
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from scipy.stats import loguniform

# --- Configurações ---
SEED = 42
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Funções Auxiliares ---

def carregar_base_openml(openml_id):
    """Carrega um dataset do OpenML e prepara os tipos de dados."""
    dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_qualities=True,
                                          download_features_meta_data=True)
    X, y, _, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

    if isinstance(y, pd.Series):
        y = y.values.ravel()

    categorical_cols_names = [name for i, name in enumerate(attribute_names) if
                              dataset.features[i].data_type == 'nominal']
    for col_name in categorical_cols_names:
        if col_name in X.columns:
            X[col_name] = X[col_name].astype('category')

    return X, y


def tunar_gbdt(modelo_nome, X_train_df, y_train_enc):
    """Executa RandomizedSearchCV para os modelos GBDT."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    if modelo_nome == 'lightgbm':
        clf = LGBMClassifier(random_state=SEED)
        param_dist = {'n_estimators': [200, 500], 'learning_rate': loguniform(0.01, 0.2), 'num_leaves': [31, 50, 70]}
    elif modelo_nome == 'xgboost':
        clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='mlogloss', tree_method='hist',
                            enable_categorical=True)
        param_dist = {'n_estimators': [200, 500], 'learning_rate': loguniform(0.01, 0.2), 'max_depth': [3, 5, 7]}
    elif modelo_nome == 'catboost':
        clf = CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)
        param_dist = {'iterations': [200, 500], 'learning_rate': loguniform(0.01, 0.2), 'depth': [4, 6, 8]}

    rand_search = RandomizedSearchCV(
        estimator=clf, param_distributions=param_dist, n_iter=15, cv=cv,
        scoring='accuracy', n_jobs=-1, random_state=SEED
    )
    rand_search.fit(X_train_df, y_train_enc)

    return rand_search.best_estimator_, rand_search.best_params_, rand_search.best_score_


# --- Função Principal de Avaliação ---

def avalia_modelo(openml_id, modelo_nome):
    """Função final que executa o fluxo correto para cada modelo."""

    X_df, y_orig = carregar_base_openml(openml_id)

    X_train_df, X_test_df, y_train_orig, y_test_orig = train_test_split(
        X_df, y_orig, test_size=0.3, random_state=SEED, stratify=(y_orig if len(np.unique(y_orig)) > 1 else None)
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_orig)
    y_test_enc = le.transform(y_test_orig)

    num_cols = X_train_df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0 and X_train_df[num_cols].isnull().sum().sum() > 0:
        imp = SimpleImputer(strategy='median')
        X_train_df.loc[:, num_cols] = imp.fit_transform(X_train_df[num_cols])
        X_test_df.loc[:, num_cols] = imp.transform(X_test_df[num_cols])

    t0_total = time.time()
    info_cv = {}

    if modelo_nome.lower() == 'realmlp':
        # Para o RealMLP, apenas instanciamos e treinamos. SEM CV externo.
        clf_final = Standalone_RealMLP_TD_S_Classifier(random_state=SEED)
        clf_final.fit(X_train_df, y_train_enc)

    elif modelo_nome.lower() in ['lightgbm', 'xgboost', 'catboost']:
        # Para os GBDTs, fazemos a busca de hiperparâmetros.
        clf_final, best_params, best_score = tunar_gbdt(modelo_nome.lower(), X_train_df, y_train_enc)
        info_cv = {'best_params': best_params, 'best_score_cv': best_score}
    else:
        raise ValueError("Modelo não suportado.")

    y_proba = clf_final.predict_proba(X_test_df)
    y_pred = clf_final.predict(X_test_df)
    total_time = time.time() - t0_total

    n_classes = len(le.classes_)
    auc_ovo = roc_auc_score(y_test_enc, y_proba, multi_class='ovo') if n_classes > 2 else roc_auc_score(y_test_enc,
                                                                                                        y_proba[:, 1])
    acc = accuracy_score(y_test_enc, y_pred)
    ce = log_loss(y_test_enc, y_proba, labels=le.transform(le.classes_))

    return {
        'openml_id': openml_id, 'modelo': modelo_nome, 'total_time_sec': total_time,
        'mean_auc_ovo': auc_ovo, 'mean_accuracy': acc, 'mean_cross_entropy': ce, **info_cv
    }


# --- Bloco de Execução ---

if __name__ == '__main__':
    cc18_ids = [11, 15, 18, 23, 29, 31, 37, 50]  # Lista reduzida para um teste rápido
    modelos = ['realmlp', 'lightgbm', 'xgboost', 'catboost']
    todos_resultados = []

    for oid in cc18_ids:
        for m in modelos:
            print(f"--- Avaliando {m} no dataset {oid} ---")
            try:
                res = avalia_modelo(openml_id=oid, modelo_nome=m)
                todos_resultados.append(res)
            except Exception as e:
                print(f"Erro em {m} @ {oid}: {e}")
                import traceback

                traceback.print_exc()
                continue

    df_all = pd.DataFrame(todos_resultados)
    print("\n--- RESULTADOS FINAIS ---")
    print(df_all)
    df_all.to_csv("resultados_finais.csv", index=False)