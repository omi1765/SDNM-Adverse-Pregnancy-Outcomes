import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocessing import load_imbalanced_data
from dnm import DNMClassifier


def get_models(input_dim):
    return {
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
        ),
        "DNM": DNMClassifier(
            input_dim=input_dim,
            n_branches=6,
            lr=0.042233,
            epochs=493
        )
    }


def run_imbalanced_experiment(csv_path, target):
    X, y = load_imbalanced_data(csv_path, target)

    X = X.values
    y = y.values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = []

    for name, model in get_models(X.shape[1]).items():
        aucs, pr_aucs = [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)

            aucs.append(roc_auc_score(y_test, y_prob))
            pr_aucs.append(average_precision_score(y_test, y_prob))

        results.append({
            "Model": name,
            "ROC_AUC": np.mean(aucs),
            "PR_AUC": np.mean(pr_aucs)
        })

    return pd.DataFrame(results)
