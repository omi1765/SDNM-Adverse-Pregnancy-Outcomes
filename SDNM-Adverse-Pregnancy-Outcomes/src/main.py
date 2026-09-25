
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import roc_curve, auc

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# =====================================================
# CONFIGURATION
# =====================================================
DATA_PATH      = "data/Mdata.csv"       # raw imbalanced data
TARGET_COLUMN  = "composite_apo"
OUTPUT_DIR     = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS       = 10
RANDOM_STATE   = 42
CTGAN_EPOCHS   = 300

# SDNM hyperparameters (from Bayesian optimization)
SDNM_N_BRANCHES = 4
SDNM_LR         = 0.042
SDNM_EPOCHS     = 493
SDNM_BETA1      = 0.90
SDNM_BETA2      = 0.999
SDNM_EPS        = 1e-8

# DNM baseline hyperparameters
DNM_N_BRANCHES  = 5
DNM_LR          = 0.01
DNM_EPOCHS      = 600


# =====================================================
# 1. LOAD AND PREPROCESS DATA
# =====================================================
def load_and_preprocess(path: str, target: str):
    """Load dataset and perform basic preprocessing."""
    df = pd.read_csv(path)

    # Drop non-informative columns if present
    drop_cols = [c for c in ["Admission number ", "Date of admission"]
                 if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df[target].values
    X = df.drop(columns=[target])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).values

    return X, y


# =====================================================
# 2. BASELINE DNM
# =====================================================
class DNM:
    """Classic Dendritic Neural Model (baseline)."""

    def __init__(self, n_branches=5, lr=0.01, epochs=600):
        self.n_branches = n_branches
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.random.randn(self.n_branches, d) * 0.1
        self.b = 0.0

        for _ in range(self.epochs):
            prod = np.prod(
                X[:, None, :] * self.W[None, :, :] + 1e-6, axis=2
            )
            z = prod.sum(axis=1) + self.b
            y_hat = self._sigmoid(z)
            error = y_hat - y

            grad_b = np.mean(error)
            grad_W = np.zeros_like(self.W)
            for i in range(self.n_branches):
                temp = (prod[:, i][:, None] / (X * self.W[i] + 1e-6)) * X
                grad_W[i] = np.mean(error[:, None] * temp, axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        prod = np.prod(
            X[:, None, :] * self.W[None, :, :] + 1e-6, axis=2
        )
        z = prod.sum(axis=1) + self.b
        p = self._sigmoid(z)
        return np.column_stack([1 - p, p])


# =====================================================
# 3. PROPOSED SDNM
# =====================================================
class SDNM:
    """
    Stable Dendritic Neural Model.
    Log-stabilized dendritic integration + linear somatic
    shortcut + Adam optimization.
    """

    def __init__(self, input_dim,
                 n_branches=4, lr=0.042, epochs=493,
                 beta1=0.90, beta2=0.999, eps=1e-8):
        self.n_branches = n_branches
        self.lr = lr
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Dendritic weights
        self.Wd = np.random.randn(n_branches, input_dim) * 0.1
        # Linear somatic shortcut
        self.Wl = np.random.randn(input_dim) * 0.1
        # Bias
        self.b = 0.0

        # Adam moment vectors
        self.mWd = np.zeros_like(self.Wd); self.vWd = np.zeros_like(self.Wd)
        self.mWl = np.zeros_like(self.Wl); self.vWl = np.zeros_like(self.Wl)
        self.mb, self.vb = 0.0, 0.0
        self.t = 0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X):
        eps = 1e-6
        # Log-stabilized dendritic integration
        mult = np.abs(X[:, None, :] * self.Wd[None, :, :]) + eps
        log_prod = np.sum(np.log(mult), axis=2)
        dendritic = np.exp(np.clip(log_prod, -500, 500))
        # Membrane: dendritic sum + linear shortcut + bias
        soma = dendritic.sum(axis=1) + X @ self.Wl + self.b
        return self._sigmoid(soma), dendritic

    def _adam(self, param, grad, m, v):
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param, m, v

    def fit(self, X, y):
        n = len(y)
        for _ in range(self.epochs):
            pred, dendritic = self._forward(X)
            delta = (pred - y) * pred * (1 - pred)

            # Gradients
            gWl = (X.T @ delta) / n
            gb = delta.mean()
            gWd = np.zeros_like(self.Wd)
            for b in range(self.n_branches):
                grad = (delta[:, None] * dendritic[:, b:b + 1]) / \
                       (X * self.Wd[b] + 1e-6)
                gWd[b] = grad.mean(axis=0)

            # Adam updates
            self.Wl, self.mWl, self.vWl = self._adam(
                self.Wl, gWl, self.mWl, self.vWl)
            self.b, self.mb, self.vb = self._adam(
                self.b, gb, self.mb, self.vb)
            self.Wd, self.mWd, self.vWd = self._adam(
                self.Wd, gWd, self.mWd, self.vWd)
        return self

    def predict_proba(self, X):
        p, _ = self._forward(X)
        return np.column_stack([1 - p, p])


# =====================================================
# 4. FOLD-WISE CTGAN AUGMENTATION
# =====================================================
def ctgan_augment_minority(X_train, y_train, epochs=300,
                           random_state=42):
    """
    Train CTGAN on minority-class samples of the training
    partition and generate synthetic samples to balance
    the training set.
    """
    minority_mask = y_train == 1
    X_minor = X_train[minority_mask]
    n_major = int((y_train == 0).sum())
    n_minor = int(minority_mask.sum())
    n_synth = n_major - n_minor

    if n_synth <= 0 or n_minor < 10:
        return X_train, y_train

    # Convert to DataFrame for SDV
    cols = [f"f{i}" for i in range(X_minor.shape[1])]
    df_minor = pd.DataFrame(X_minor, columns=cols)

    # Build metadata and train CTGAN
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_minor)

    ctgan = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        verbose=False,
    )
    ctgan.fit(df_minor)

    # Generate synthetic samples
    df_synth = ctgan.sample(num_rows=n_synth)
    X_synth = df_synth[cols].values

    # Combine
    X_balanced = np.vstack([X_train, X_synth])
    y_balanced = np.concatenate([
        y_train, np.ones(len(X_synth), dtype=int)
    ])
    return X_balanced, y_balanced


# =====================================================
# 5. MAIN EXPERIMENT — 10-FOLD CV WITH FOLD-WISE CTGAN
# =====================================================
def main():
    print("=" * 60)
    print("SDNM Adverse Pregnancy Outcome Prediction")
    print("Fold-wise CTGAN + 10-fold Stratified CV")
    print("=" * 60)

    # Load data
    X, y = load_and_preprocess(DATA_PATH, TARGET_COLUMN)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Classifier dictionary
    classifiers = {
        "DT":   DecisionTreeClassifier(
                    max_depth=2, min_samples_split=10,
                    random_state=RANDOM_STATE),
        "RF":   RandomForestClassifier(
                    n_estimators=5, max_depth=2,
                    min_samples_split=5,
                    random_state=RANDOM_STATE),
        "GBM":  GradientBoostingClassifier(
                    n_estimators=5, learning_rate=0.05,
                    max_depth=2, random_state=RANDOM_STATE),
        "XGBM": XGBClassifier(
                    n_estimators=5, max_depth=2,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE),
        "LGBM": LGBMClassifier(
                    n_estimators=5, num_leaves=4,
                    learning_rate=0.05, max_depth=2,
                    random_state=RANDOM_STATE, verbose=-1),
    }

    # Storage for predictions
    predictions = {m: [] for m in classifiers}
    predictions["DNM"]  = []
    predictions["SDNM"] = []
    true_labels = []

    # 10-fold stratified CV
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True,
        random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        X_train_raw = X[train_idx]
        X_test      = X[test_idx]
        y_train_raw = y[train_idx]
        y_test      = y[test_idx]

        # ---- Fold-wise CTGAN augmentation ----
        print("  Training CTGAN on minority class...")
        X_train, y_train = ctgan_augment_minority(
            X_train_raw, y_train_raw,
            epochs=CTGAN_EPOCHS,
            random_state=RANDOM_STATE + fold)
        print(f"  Training set after CTGAN: {X_train.shape}, "
              f"class balance: {np.bincount(y_train)}")

        true_labels.extend(y_test)

        # ---- Tree-based models ----
        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            predictions[name].extend(prob)

        # ---- Scaling for neural models ----
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        # ---- DNM baseline ----
        dnm = DNM(
            n_branches=DNM_N_BRANCHES,
            lr=DNM_LR,
            epochs=DNM_EPOCHS,
        )
        dnm.fit(X_tr, y_train)
        predictions["DNM"].extend(
            dnm.predict_proba(X_te)[:, 1])

        # ---- Proposed SDNM ----
        sdnm = SDNM(
            input_dim=X.shape[1],
            n_branches=SDNM_N_BRANCHES,
            lr=SDNM_LR,
            epochs=SDNM_EPOCHS,
            beta1=SDNM_BETA1,
            beta2=SDNM_BETA2,
            eps=SDNM_EPS,
        )
        sdnm.fit(X_tr, y_train)
        predictions["SDNM"].extend(
            sdnm.predict_proba(X_te)[:, 1])

    # =================================================
    # 6. ROC-AUC PLOT
    # =================================================
    print("\n" + "=" * 60)
    print("Generating ROC-AUC comparison plot...")
    print("=" * 60)

    colors = {
        "DT":   "blue",
        "RF":   "green",
        "GBM":  "orange",
        "XGBM": "purple",
        "LGBM": "brown",
        "DNM":  "black",
        "SDNM": "red",
    }

    plt.figure(figsize=(10, 8), dpi=600)

    for name, prob in predictions.items():
        fpr, tpr, _ = roc_curve(true_labels, prob)
        roc_auc = auc(fpr, tpr)

        if name == "SDNM":
            plt.plot(fpr, tpr,
                     color=colors[name],
                     linewidth=4,
                     label=f"{name} (AUC={roc_auc:.4f})",
                     zorder=10)
        else:
            plt.plot(fpr, tpr,
                     color=colors[name],
                     linewidth=2.2,
                     label=f"{name} (AUC={roc_auc:.4f})",
                     alpha=0.85)

    # Random classifier
    plt.plot([0, 1], [0, 1],
             linestyle="--", color="gray",
             linewidth=1.5, label="Random Guess")

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("False Positive Rate",
               fontsize=16, fontweight="bold")
    plt.ylabel("True Positive Rate",
               fontsize=16, fontweight="bold")
    plt.title("ROC-AUC Curve Comparison",
              fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="lower right", frameon=True)
    plt.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()

    
