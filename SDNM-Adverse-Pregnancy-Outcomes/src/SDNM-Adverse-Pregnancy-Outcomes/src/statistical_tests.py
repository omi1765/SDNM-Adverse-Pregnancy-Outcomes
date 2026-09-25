import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats
from itertools import combinations

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata



# =====================================================
# PUBLICATION STYLE
# =====================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "legend.frameon": True,
    "axes.linewidth": 1.2,
    "savefig.dpi": 600,
    "figure.dpi": 600,
})


# =====================================================
# LOAD DATA
# =====================================================
def load_and_preprocess(path, target):
    df = pd.read_csv(path)
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
# DNM & SDNM (match main.py)
# =====================================================
class DNM:
    def __init__(self, n_branches=5, lr=0.01, epochs=600):
        self.n_branches, self.lr, self.epochs = n_branches, lr, epochs

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.random.randn(self.n_branches, d) * 0.1
        self.b = 0.0
        for _ in range(self.epochs):
            prod = np.prod(
                X[:, None, :] * self.W[None, :, :] + 1e-6, axis=2)
            z = prod.sum(axis=1) + self.b
            y_hat = self._sigmoid(z)
            error = y_hat - y
            grad_b = np.mean(error)
            grad_W = np.zeros_like(self.W)
            for i in range(self.n_branches):
                temp = (prod[:, i][:, None] /
                        (X * self.W[i] + 1e-6)) * X
                grad_W[i] = np.mean(error[:, None] * temp, axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        prod = np.prod(
            X[:, None, :] * self.W[None, :, :] + 1e-6, axis=2)
        z = prod.sum(axis=1) + self.b
        p = self._sigmoid(z)
        return np.column_stack([1 - p, p])


class SDNM:
    def __init__(self, input_dim, n_branches=4, lr=0.042, epochs=493,
                 beta1=0.90, beta2=0.999, eps=1e-8):
        self.n_branches, self.lr, self.epochs = n_branches, lr, epochs
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.Wd = np.random.randn(n_branches, input_dim) * 0.1
        self.Wl = np.random.randn(input_dim) * 0.1
        self.b = 0.0
        self.mWd = np.zeros_like(self.Wd); self.vWd = np.zeros_like(self.Wd)
        self.mWl = np.zeros_like(self.Wl); self.vWl = np.zeros_like(self.Wl)
        self.mb, self.vb, self.t = 0, 0, 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X):
        eps = 1e-6
        mult = np.abs(X[:, None, :] * self.Wd[None, :, :]) + eps
        log_prod = np.sum(np.log(mult), axis=2)
        dendritic = np.exp(np.clip(log_prod, -500, 500))
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
            gWl = (X.T @ delta) / n
            gb = delta.mean()
            gWd = np.zeros_like(self.Wd)
            for b in range(self.n_branches):
                grad = (delta[:, None] * dendritic[:, b:b + 1]) / \
                       (X * self.Wd[b] + 1e-6)
                gWd[b] = grad.mean(axis=0)
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
# FOLD-WISE CTGAN (matches main.py)
# =====================================================
def ctgan_augment_minority(X_train, y_train, epochs=300,
                           random_state=42):
    minority_mask = y_train == 1
    X_minor = X_train[minority_mask]
    n_major = int((y_train == 0).sum())
    n_minor = int(minority_mask.sum())
    n_synth = n_major - n_minor

    if n_synth <= 0 or n_minor < 10:
        return X_train, y_train

    cols = [f"f{i}" for i in range(X_minor.shape[1])]
    df_minor = pd.DataFrame(X_minor, columns=cols)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_minor)
    ctgan = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    ctgan.fit(df_minor)
    df_synth = ctgan.sample(num_rows=n_synth)
    X_synth = df_synth[cols].values

    X_balanced = np.vstack([X_train, X_synth])
    y_balanced = np.concatenate(
        [y_train, np.ones(len(X_synth), dtype=int)])
    return X_balanced, y_balanced


# =====================================================
# TEST 1: FRIEDMAN
# =====================================================
def friedman_test(fold_metrics, metric="AUC", models=None):
    M = np.array([fold_metrics[m][metric] for m in models]).T
    stat, p = stats.friedmanchisquare(
        *[M[:, i] for i in range(M.shape[1])])
    return stat, p


# =====================================================
# TEST 2: NEMENYI
# =====================================================
def nemenyi_posthoc(fold_metrics, metric="AUC", models=None,
                    alpha=0.05):
    M = np.array([fold_metrics[m][metric] for m in models]).T
    n_folds, k = M.shape

    ranks = np.zeros_like(M)
    for i in range(n_folds):
        ranks[i] = stats.rankdata(-M[i])
    mean_ranks = ranks.mean(axis=0)

    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table[k]
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_folds))

    sig_pairs = []
    for i, j in combinations(range(k), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        sig_pairs.append((models[i], models[j], diff, diff > CD))

    return mean_ranks, CD, sig_pairs


# =====================================================
# TEST 3: WILCOXON (with Holm-Bonferroni)
# =====================================================
def wilcoxon_vs_sdnm(fold_metrics, metric="AUC",
                     reference="SDNM", models=None):
    ref_vals = np.array(fold_metrics[reference][metric])
    baselines = [m for m in models if m != reference]

    raw_pvals, stats_list = [], []
    for m in baselines:
        base_vals = np.array(fold_metrics[m][metric])
        try:
            stat, p = stats.wilcoxon(
                ref_vals, base_vals, alternative="greater")
        except ValueError:
            stat, p = 0.0, 1.0
        stats_list.append(stat)
        raw_pvals.append(p)

    # Holm–Bonferroni correction
    raw_pvals = np.array(raw_pvals)
    order = np.argsort(raw_pvals)
    n = len(raw_pvals)
    holm_pvals = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(raw_pvals[idx] * (n - rank), 1.0)
        running_max = max(running_max, adjusted)
        holm_pvals[idx] = running_max

    return baselines, stats_list, raw_pvals, holm_pvals


# =====================================================
# TEST 4: McNEMAR
# =====================================================
def mcnemar_test(y_true_all, pred_a, pred_b, exact=False):
    y_true_all = np.asarray(y_true_all)
    correct_a = (pred_a == y_true_all).astype(int)
    correct_b = (pred_b == y_true_all).astype(int)

    n00 = np.sum((correct_a == 0) & (correct_b == 0))
    n01 = np.sum((correct_a == 0) & (correct_b == 1))
    n10 = np.sum((correct_a == 1) & (correct_b == 0))
    n11 = np.sum((correct_a == 1) & (correct_b == 1))

    b, c = n01, n10

    if exact or (b + c) < 25:
        from scipy.stats import binomtest
        result = binomtest(b, b + c, 0.5)
        p = result.pvalue
        stat = np.nan
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = 1 - stats.chi2.cdf(stat, df=1)

    return stat, p, (n00, n01, n10, n11)


def concat_fold_data(fold_predictions, model_name):
    y_all, prob_all, label_all = [], [], []
    for yt, pr, lb in fold_predictions[model_name]:
        y_all.append(yt); prob_all.append(pr); label_all.append(lb)
    return (np.concatenate(y_all),
            np.concatenate(prob_all),
            np.concatenate(label_all))


# =====================================================
# TEST 5: DeLONG
# =====================================================
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, prob_a, prob_b):
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    preds = np.vstack([prob_a[order], prob_b[order]])
    aucs, cov = _fast_delong(preds, label_1_count)

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p


# =====================================================
# CRITICAL DIFFERENCE DIAGRAM
# =====================================================
def plot_critical_difference(fold_metrics, metric="AUC",
                             models=None, alpha=0.05,
                             filename="CriticalDifferenceDiagram.png"):
    M = np.array([fold_metrics[m][metric] for m in models]).T
    n_folds, k = M.shape

    ranks = np.zeros_like(M)
    for i in range(n_folds):
        ranks[i] = stats.rankdata(-M[i])
    mean_ranks = ranks.mean(axis=0)

    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table[k]
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_folds))

    order = np.argsort(mean_ranks)
    sorted_names = [models[i] for i in order]
    sorted_ranks = mean_ranks[order]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=600)
    y_axis = 1.0
    ax.plot([1, k], [y_axis, y_axis], color="black", linewidth=1.5)

    for i in range(1, k + 1):
        ax.plot([i, i], [y_axis - 0.02, y_axis + 0.02],
                color="black", linewidth=1.0)
        ax.text(i, y_axis + 0.05, str(i),
                ha="center", va="bottom", fontsize=11)

    for i, (name, r) in enumerate(zip(sorted_names, sorted_ranks)):
        side = 1 if i % 2 == 0 else -1
        y_off = 0.15 * side
        ax.plot([r, r], [y_axis, y_axis + y_off * 0.8],
                color="black", linewidth=0.8)
        ax.text(r, y_axis + y_off,
                f"{name} ({r:.2f})",
                ha="center",
                va="bottom" if side > 0 else "top",
                fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#d62728" if name == "SDNM" else "lightgray",
                          edgecolor="black", linewidth=1.0, alpha=0.9))

    best_rank = sorted_ranks[0]
    ax.plot([best_rank, best_rank + CD], [y_axis - 0.15, y_axis - 0.15],
            color="black", linewidth=3.0)
    ax.text(best_rank + CD / 2, y_axis - 0.22,
            f"CD = {CD:.2f}", ha="center", va="top",
            fontsize=10.5, fontweight="bold")

    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] <= CD:
                y_line = y_axis - 0.06 - 0.03 * (j - i)
                ax.plot([sorted_ranks[i], sorted_ranks[j]],
                        [y_line, y_line],
                        color="steelblue", linewidth=4,
                        alpha=0.45, solid_capstyle="round")

    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(y_axis - 0.4, y_axis + 0.35)
    ax.set_yticks([])
    ax.set_xlabel(f"Average Rank ({metric})",
                  fontsize=12, fontweight="bold")
    ax.set_title(f"Nemenyi Critical Difference Diagram — {metric}",
                 fontsize=13, fontweight="bold", pad=20)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(str(out).replace(".png", ".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 60)
    print("Statistical Testing Suite")
    print("Fold-wise CTGAN + 10-fold Stratified CV")
    print("=" * 60)

    X, y = load_and_preprocess(DATA_PATH, TARGET_COLUMN)
    print(f"Dataset: {X.shape}, classes: {np.bincount(y)}")

    models = {
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
    MODEL_NAMES = ["DT", "RF", "GBM", "XGBM", "LGBM", "DNM", "SDNM"]

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True,
        random_state=RANDOM_STATE)

    fold_metrics = {m: {"ACC": [], "F1": [], "MCC": [], "AUC": []}
                    for m in MODEL_NAMES}
    fold_predictions = {m: [] for m in MODEL_NAMES}

    for fold, (train, test) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/{N_SPLITS}")
        X_train_raw, X_test = X[train], X[test]
        y_train_raw, y_test = y[train], y[test]

        # Fold-wise CTGAN
        X_train, y_train = ctgan_augment_minority(
            X_train_raw, y_train_raw,
            epochs=CTGAN_EPOCHS,
            random_state=RANDOM_STATE + fold)

        # Tree-based models
        for name, model in models.items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            label = (prob >= 0.5).astype(int)

            fold_metrics[name]["ACC"].append(
                accuracy_score(y_test, label))
            fold_metrics[name]["F1"].append(
                f1_score(y_test, label, zero_division=0))
            fold_metrics[name]["MCC"].append(
                matthews_corrcoef(y_test, label))
            fold_metrics[name]["AUC"].append(
                roc_auc_score(y_test, prob))
            fold_predictions[name].append((y_test, prob, label))

        # Scaled neural inputs
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        for name, model in [
            ("DNM",  DNM(n_branches=DNM_N_BRANCHES,
                         lr=DNM_LR, epochs=DNM_EPOCHS)),
            ("SDNM", SDNM(input_dim=X.shape[1],
                          n_branches=SDNM_N_BRANCHES,
                          lr=SDNM_LR, epochs=SDNM_EPOCHS,
                          beta1=SDNM_BETA1, beta2=SDNM_BETA2,
                          eps=SDNM_EPS)),
        ]:
            model.fit(Xtr, y_train)
            prob = model.predict_proba(Xte)[:, 1]
            label = (prob >= 0.5).astype(int)

            fold_metrics[name]["ACC"].append(
                accuracy_score(y_test, label))
            fold_metrics[name]["F1"].append(
                f1_score(y_test, label, zero_division=0))
            fold_metrics[name]["MCC"].append(
                matthews_corrcoef(y_test, label))
            fold_metrics[name]["AUC"].append(
                roc_auc_score(y_test, prob))
            fold_predictions[name].append((y_test, prob, label))

    # =================================================
    # TEST 1: FRIEDMAN
    # =================================================
    print("\n" + "=" * 75)
    print("FRIEDMAN TEST (global)")
    print("=" * 75)
    for metric in ["AUC", "F1", "ACC", "MCC"]:
        stat, p = friedman_test(fold_metrics, metric, MODEL_NAMES)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else \
              "*" if p < 0.05 else "ns"
        print(f"{metric:>5}:  χ² = {stat:8.4f}   "
              f"p = {p:.4e}   {sig}")

    # =================================================
    # TEST 2: NEMENYI
    # =================================================
    print("\n" + "=" * 75)
    print("NEMENYI POST-HOC")
    print("=" * 75)
    for metric in ["AUC", "F1"]:
        mean_ranks, CD, sig_pairs = nemenyi_posthoc(
            fold_metrics, metric, MODEL_NAMES)
        print(f"\n--- Metric: {metric} ---")
        for idx in np.argsort(mean_ranks):
            print(f"  {MODEL_NAMES[idx]:>5}: "
                  f"mean rank = {mean_ranks[idx]:.3f}")
        print(f"  Critical Difference (CD) = {CD:.4f}")

    # =================================================
    # TEST 3: WILCOXON
    # =================================================
    print("\n" + "=" * 75)
    print("WILCOXON SIGNED-RANK (SDNM vs each baseline)")
    print("=" * 75)
    for metric in ["AUC", "F1", "ACC", "MCC"]:
        baselines, W, raw, holm = wilcoxon_vs_sdnm(
            fold_metrics, metric, "SDNM", MODEL_NAMES)
        print(f"\n--- Metric: {metric} ---")
        print(f"{'Comparison':<20}{'W':>8}"
              f"{'p (raw)':>14}{'p (Holm)':>14}{'Sig':>6}")
        for b, w, p, ph in zip(baselines, W, raw, holm):
            sig = "***" if ph < 0.001 else \
                  "**" if ph < 0.01 else \
                  "*" if ph < 0.05 else "ns"
            print(f"SDNM vs {b:<10}{w:>8.1f}"
                  f"{p:>14.4e}{ph:>14.4e}{sig:>6}")

    # =================================================
    # TEST 4: McNEMAR
    # =================================================
    y_true_all, _, sdnm_label = concat_fold_data(
        fold_predictions, "SDNM")

    print("\n" + "=" * 75)
    print("McNEMAR'S TEST")
    print("=" * 75)
    print(f"{'Comparison':<20}{'χ²':>10}{'p-value':>16}{'Sig':>6}")
    for m in MODEL_NAMES:
        if m == "SDNM":
            continue
        _, _, base_label = concat_fold_data(fold_predictions, m)
        stat, p, _ = mcnemar_test(
            y_true_all, sdnm_label, base_label)
        sig = "***" if p < 0.001 else \
              "**" if p < 0.01 else \
              "*" if p < 0.05 else "ns"
        stat_str = f"{stat:.3f}" if not np.isnan(stat) else "exact"
        print(f"SDNM vs {m:<10}{stat_str:>10}"
              f"{p:>16.4e}{sig:>6}")

    # =================================================
    # TEST 5: DeLONG
    # =================================================
    _, sdnm_prob, _ = concat_fold_data(fold_predictions, "SDNM")

    print("\n" + "=" * 75)
    print("DeLONG TEST (AUC comparison)")
    print("=" * 75)
    print(f"{'Comparison':<20}{'AUC_SDNM':>12}"
          f"{'AUC_base':>12}{'Z':>10}{'p':>14}{'Sig':>6}")
    for m in MODEL_NAMES:
        if m == "SDNM":
            continue
        _, base_prob, _ = concat_fold_data(fold_predictions, m)
        auc_a, auc_b, z, p = delong_test(
            y_true_all, sdnm_prob, base_prob)
        sig = "***" if p < 0.001 else \
              "**" if p < 0.01 else \
              "*" if p < 0.05 else "ns"
        print(f"SDNM vs {m:<10}{auc_a:>12.4f}"
              f"{auc_b:>12.4f}{z:>10.3f}{p:>14.4e}{sig:>6}")

    # =================================================
    # SAVE SUMMARY CSV
    # =================================================
    summary_rows = []
    for metric in ["ACC", "F1", "MCC", "AUC"]:
        row = {"Metric": metric}
        for m in MODEL_NAMES:
            vals = np.array(fold_metrics[m][metric])
            row[m] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUTPUT_DIR / "statistical_test_summary.csv", index=False)
    print(f"\nSaved: "
          f"{OUTPUT_DIR / 'statistical_test_summary.csv'}")

    # =================================================
    # CRITICAL DIFFERENCE DIAGRAMS
    # =================================================
    plot_critical_difference(
        fold_metrics, metric="AUC", models=MODEL_NAMES,
        filename="CriticalDifferenceDiagram_AUC.png")
    plot_critical_difference(
        fold_metrics, metric="F1", models=MODEL_NAMES,
        filename="CriticalDifferenceDiagram_F1.png")

    print("\nDone.")
