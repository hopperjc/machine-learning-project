import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def carregar_base_openml(openml_id, test_size=0.3, seed=42):
    dataset = openml.datasets.get_dataset(openml_id, download_all_files=False)
    X_df, y, categorial_indicator, feat_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1,)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y))>1 else None
    )
    return X_train_df, X_test_df, y_train, y_test