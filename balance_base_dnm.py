import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv("Mdata_CTGAN.csv")
target = "composite_apo"

df = df.drop(columns=[c for c in ["Admission number ", "Date of admission"] if c in df.columns])

y = df[target].values
X = df.drop(columns=[target])

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).values

print("Balanced class distribution:")
print(pd.Series(y).value_counts())


models = {
    "DT": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RF": RandomForestClassifier(n_estimators=300, random_state=42),
    "GBM": GradientBoostingClassifier(n_estimators=300, random_state=42),
    "XGBM": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ),
    "LGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
}


class DNM:
    def __init__(self, input_dim, n_branches=6, lr=0.01, epochs=300):
        self.input_dim = input_dim
        self.n_branches = n_branches
        self.lr = lr
        self.epochs = epochs
        self.W = np.random.randn(n_branches, input_dim) * 0.1
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        mult = X[:, None, :] * self.W[None, :, :]
        dend = np.prod(mult + 1e-6, axis=2)
        soma = np.sum(dend, axis=1) + self.b
        return self.sigmoid(soma)

    def fit(self, X, y):
        for _ in range(self.epochs):
            yhat = self.forward(X)
            error = yhat - y
            self.b -= self.lr * np.mean(error)
            for j in range(self.n_branches):
                grad = np.mean(error[:, None] / (X + 1e-6), axis=0)
                self.W[j] -= self.lr * grad

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ============================
# ADD DNM
# ============================
models["DNM"] = DNM(input_dim=X.shape[1])


