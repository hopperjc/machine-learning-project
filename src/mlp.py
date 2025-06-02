# mlp_no_preproc.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import TensorDataset, DataLoader


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # camada de escala aprendível (inicializada com vetores de 1)
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        # peso e bias inicializados normalmente ou zerados
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # projeta e normaliza pela raiz quadrada de in_features
        return (1.0 / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # função de ativação Mish: x * tanh(softplus(x))
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))


class SimpleMLP(BaseEstimator):
    """
    Um MLP simples para classificação/regressão.
    - is_classification=True → usa CrossEntropyLoss + SELU.
    - is_classification=False → usa MSELoss + Mish.

    Supõe que X (numpy array) NÃO tenha features categóricas nem NaNs:
    X já deve estar pronto para treinar (somente floats).
    """

    def __init__(self, is_classification: bool, device: str = "cpu"):
        self.is_classification = is_classification
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        # X: numpy array (n_samples, n_features)
        # y: numpy array (n_samples,) ou (n_samples, n_outputs)
        input_dim = X.shape[1]
        is_class = self.is_classification

        # define saída:
        if is_class:
            # Encoder ordinal para converter labels em [0,1,...,K-1]
            self.class_enc_ = OrdinalEncoder(dtype=np.int64)
            y_enc = self.class_enc_.fit_transform(y.reshape(-1, 1)).ravel()
            self.classes_ = self.class_enc_.categories_[0]
            output_dim = len(self.classes_)
            y_train = y_enc
        else:
            # Regressão: normalizar targets
            self.y_mean_ = np.mean(y, axis=0)
            self.y_std_ = np.std(y, axis=0)
            y_norm = (y - self.y_mean_) / (self.y_std_ + 1e-30)
            y_train = y_norm if y_norm.ndim > 1 else y_norm.reshape(-1, 1)

            if y_val is not None:
                y_val_norm = (y_val - self.y_mean_) / (self.y_std_ + 1e-30)
                y_val = y_val_norm if y_val_norm.ndim > 1 else y_val_norm.reshape(-1, 1)

            output_dim = y_train.shape[1]

        # Monta a rede:
        act = nn.SELU if is_class else Mish
        model = nn.Sequential(
            ScalingLayer(input_dim),
            NTPLinear(input_dim, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, output_dim, zero_init=True),
        ).to(self.device)

        # Critério e otimizador
        criterion = (
            nn.CrossEntropyLoss(label_smoothing=0.1)
            if is_class
            else nn.MSELoss()
        )
        params = list(model.parameters())
        scale_p = [params[0]]
        weights = params[1::2]
        biases = params[2::2]
        optimizer = torch.optim.Adam(
            [dict(params=scale_p), dict(params=weights), dict(params=biases)],
            betas=(0.9, 0.95),
        )

        # Converte para tensores
        x_train = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.as_tensor(
            y_train,
            dtype=torch.int64 if is_class else torch.float32
        ).to(self.device)
        if not is_class and y_train_tensor.ndim == 1:
            y_train_tensor = y_train_tensor.unsqueeze(1)

        if X_val is not None and y_val is not None:
            x_valid = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
            y_valid_tensor = torch.as_tensor(
                y_val,
                dtype=torch.int64 if is_class else torch.float32
            ).to(self.device)
            if not is_class and y_valid_tensor.ndim == 1:
                y_valid_tensor = y_valid_tensor.unsqueeze(1)
        else:
            x_valid = x_train[:0]
            y_valid_tensor = y_train_tensor[:0]

        # DataLoaders
        train_ds = TensorDataset(x_train, y_train_tensor)
        valid_ds = TensorDataset(x_valid, y_valid_tensor)
        n_train = x_train.shape[0]
        n_valid = x_valid.shape[0]
        n_epochs = 256
        batch_size_train = min(256, n_train)
        batch_size_valid = max(1, min(1024, n_valid))

        train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size_valid, shuffle=False)

        # Scheduler interno de learning rate:
        def validation_metric(y_pred: torch.Tensor, y_true: torch.Tensor):
            if is_class:
                # erro de classificação sem normalizar
                return torch.sum(torch.argmax(y_pred, dim=-1) != y_true)
            else:
                # MSE 
                return (y_pred - y_true).square().mean()

        n_batches = len(train_loader)
        base_lr = 0.04 if is_class else 0.07
        best_valid_loss = np.inf
        best_params_copy = None

        # Loop de treinamento
        for epoch in range(n_epochs):
            for batch_idx, (x_b, y_b) in enumerate(train_loader):
                t = (epoch * n_batches + batch_idx) / (n_epochs * n_batches)
                sched = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
                lr = base_lr * sched
                optimizer.param_groups[0]['lr'] = 6 * lr   # scale layer
                optimizer.param_groups[1]['lr'] = lr       # weights
                optimizer.param_groups[2]['lr'] = 0.1 * lr  # biases

                y_pred = model(x_b)
                loss = criterion(y_pred, y_b)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # checar validação
            with torch.no_grad():
                if x_valid.shape[0] > 0:
                    all_preds = []
                    for xv, _ in valid_loader:
                        all_preds.append(model(xv))
                    y_pred_valid = torch.cat(all_preds, dim=0)
                    valid_loss = validation_metric(y_pred_valid, y_valid_tensor).cpu().item()
                else:
                    valid_loss = 0.0

                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss
                    # copia parâmetros
                    best_params_copy = [p.detach().clone() for p in model.parameters()]

        # Recarrega os melhores parâmetros
        with torch.no_grad():
            for p_mod, p_best in zip(model.parameters(), best_params_copy):
                p_mod.set_(p_best)

        self.model_ = model
        return self

    def predict(self, X):
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(x).cpu().numpy()
        if self.is_classification:
            # devolve as classes originais com inverse_transform
            idxs = np.argmax(out, axis=-1).reshape(-1, 1)
            return self.class_enc_.inverse_transform(idxs).ravel()
        else:
            # reescala para o domínio original
            return out[:, 0] * self.y_std_ + self.y_mean_

    def predict_proba(self, X):
        assert self.is_classification, "predict_proba só vale para classificação"
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


class Standalone_RealMLP_TD_S_Classifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper de conveniência que treina SimpleMLP diretamente sobre X numérico + y.
    (Não faz nenhuma transformação adicional.)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        # Aqui assumimos que X já está 100% numérico (sem NaNs, sem colunas categóricas)
        self.model_ = SimpleMLP(is_classification=True, device=self.device)
        self.model_.fit(X, y, X_val, y_val)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class Standalone_RealMLP_TD_S_Regressor(BaseEstimator, RegressorMixin):
    """
    Wrapper de conveniência para regressão com SimpleMLP (dados já numéricos).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        self.model_ = SimpleMLP(is_classification=False, device=self.device)
        self.model_.fit(X, y, X_val, y_val)
        return self

    def predict(self, X):
        return self.model_.predict(X)
