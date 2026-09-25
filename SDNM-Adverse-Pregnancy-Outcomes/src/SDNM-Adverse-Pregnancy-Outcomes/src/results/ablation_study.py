import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, brier_score_loss,
)
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# =====================================================
# CONFIGURATION
# =====================================================
DATA_PATH      = "data/Mdata.csv"
TARGET_COLUMN  = "composite_apo"
OUTPUT_DIR     = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS       = 10
RANDOM_STATE   = 42
CTGAN_EPOCHS   = 300


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
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
    "axes.linewidth": 1.2,
    "savefig.dpi": 600,
    "figure.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
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
# FOLD-WISE CTGAN
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
# UNIFIED ABLATION MODEL
# =====================================================
class DendriticModel:
    """
    Unified dendritic model with toggleable components.

    Parameters
    ----------
    use_log    : bool  -> log-stabilized dendritic integration
    use_linear : bool  -> linear somatic shortcut pathway
    use_adam   : bool  -> Adam optimizer (else plain GD)
    """

    def __init__(self, input_dim, n_branches=4,
                 lr=0.042, epochs=493,
                 use_log=True, use_linear=True, use_adam=True,
                 beta1=0.90, beta2=0.999, eps=1e-8, seed=42):

        rng = np.random.default_rng(seed)

        self.n_branches = n_branches
        self.lr         = lr
        self.epochs     = epochs
        self.use_log    = use_log
        self.use_linear = use_linear
        self.use_adam   = use_adam
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

        self.Wd = rng.standard_normal((n_branches, input_dim)) * 0.1
        if use_linear:
            self.Wl = rng.standard_normal(input_dim) * 0.1
        self.b = 0.0

        if use_adam:
            self.mWd = np.zeros_like(self.Wd)
            self.vWd = np.zeros_like(self.Wd)
            if use_linear:
                self.mWl = np.zeros_like(self.Wl)
                self.vWl = np.zeros_like(self.Wl)
            self.mb, self.vb, self.t = 0.0, 0.0, 0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        eps = 1e-6
        if self.use_log:
            mult = np.abs(X[:, None, :] * self.Wd[None, :, :]) + eps
            log_prod = np.sum(np.log(mult), axis=2)
            dendritic = np.exp(np.clip(log_prod, -500, 500))
        else:
            dendritic = np.prod(
                X[:, None, :] * self.Wd[None, :, :] + eps, axis=2)

        soma = dendritic.sum(axis=1)
        if self.use_linear:
            soma = soma + X @ self.Wl
        soma = soma + self.b
        return self._sigmoid(soma), dendritic

    def _adam_step(self, param, grad, m, v):
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
            pred, dendritic = self.forward(X)
            delta = (pred - y) * pred * (1 - pred)

            gb = delta.mean()
            gWd = np.zeros_like(self.Wd)
            for b in range(self.n_branches):
                grad = (delta[:, None] * dendritic[:, b:b + 1]) / \
                       (X * self.Wd[b] + 1e-6)
                gWd[b] = grad.mean(axis=0)
            gWl = (X.T @ delta) / n if self.use_linear else None

            if self.use_adam:
                self.Wd, self.mWd, self.vWd = self._adam_step(
                    self.Wd, gWd, self.mWd, self.vWd)
                self.b, self.mb, self.vb = self._adam_step(
                    self.b, gb, self.mb, self.vb)
                if self.use_linear:
                    self.Wl, self.mWl, self.vWl = self._adam_step(
                        self.Wl, gWl, self.mWl, self.vWl)
            else:
                self.Wd -= self.lr * gWd
                self.b -= self.lr * gb
                if self.use_linear:
                    self.Wl -= self.lr * gWl
        return self

    def predict_proba(self, X):
        p, _ = self.forward(X)
        return np.column_stack([1 - p, p])


# =====================================================
# ABLATION VARIANTS
# =====================================================
VARIANTS = {
    "DNM":              dict(use_log=False, use_linear=False, use_adam=False),
    "DNM+Log":          dict(use_log=True,  use_linear=False, use_adam=False),
    "DNM+Adam":         dict(use_log=False, use_linear=False, use_adam=True),
    "SDNM (GD)":        dict(use_log=True,  use_linear=True,  use_adam=False),
    "Full SDNM+Adam":   dict(use_log=True,  use_linear=True,  use_adam=True),
}
VARIANT_ORDER = ["DNM", "DNM+Log", "DNM+Adam",
                 "SDNM (GD)", "Full SDNM+Adam"]


# =====================================================
# STATISTICAL HELPERS
# =====================================================
def holm_bonferroni(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        a = min(pvals[idx] * (n - rank), 1.0)
        running = max(running, a)
        adj[idx] = running
    return adj


def nemenyi_posthoc(fold_metrics, metric, alpha=0.05):
    M = np.array([fold_metrics[v][metric] for v in VARIANT_ORDER]).T
    n_folds, k = M.shape
    ranks = np.zeros_like(M)
    for i in range(n_folds):
        ranks[i] = stats.rankdata(-M[i])
    mean_ranks = ranks.mean(axis=0)

    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                     6: 2.850, 7: 2.949, 8: 3.031,
                     9: 3.102, 10: 3.164}
    q_alpha = q_alpha_table[k]
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_folds))
    return mean_ranks, CD


# =====================================================
# CRITICAL DIFFERENCE DIAGRAM
# =====================================================
def plot_critical_difference(fold_metrics, metric="AUC",
                             filename="Ablation_CD_Diagram.png"):
    mr, CD = nemenyi_posthoc(fold_metrics, metric)
    order = np.argsort(mr)
    names = [VARIANT_ORDER[i] for i in order]
    ranks = mr[order]
    k = len(names)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=600)
    y_axis = 1.0
    ax.plot([1, k], [y_axis, y_axis], color="black", linewidth=1.5)

    for i in range(1, k + 1):
        ax.plot([i, i], [y_axis - 0.02, y_axis + 0.02],
                color="black", linewidth=1.0)
        ax.text(i, y_axis + 0.05, str(i),
                ha="center", va="bottom", fontsize=11)

    for i, (name, r) in enumerate(zip(names, ranks)):
        side = 1 if i % 2 == 0 else -1
        y_off = 0.15 * side
        ax.plot([r, r], [y_axis, y_axis + y_off * 0.8],
                color="black", linewidth=0.8)
        face = "#d62728" if name == "Full SDNM+Adam" else "lightgray"
        ax.text(r, y_axis + y_off, f"{name} ({r:.2f})",
                ha="center",
                va="bottom" if side > 0 else "top",
                fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=face, edgecolor="black",
                          linewidth=1.0, alpha=0.9))

    best = ranks[0]
    ax.plot([best, best + CD], [y_axis - 0.18, y_axis - 0.18],
            color="black", linewidth=3.0)
    ax.text(best + CD / 2, y_axis - 0.26,
            f"CD = {CD:.2f}", ha="center", va="top",
            fontsize=10.5, fontweight="bold")

    for i in range(k):
        for j in range(i + 1, k):
            if ranks[j] - ranks[i] <= CD:
                y_line = y_axis - 0.08 - 0.03 * (j - i)
                ax.plot([ranks[i], ranks[j]], [y_line, y_line],
                        color="steelblue", linewidth=4,
                        alpha=0.45, solid_capstyle="round")

    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(y_axis - 0.45, y_axis + 0.35)
    ax.set_yticks([])
    ax.set_xlabel(f"Average Rank ({metric})",
                  fontsize=12, fontweight="bold")
    ax.set_title(f"Nemenyi Critical Difference Diagram — Ablation ({metric})",
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
    print("Ablation Study — SDNM Components")
    print("Fold-wise CTGAN + 10-fold Stratified CV")
    print("=" * 60)

    X, y = load_and_preprocess(DATA_PATH, TARGET_COLUMN)
    print(f"Dataset: {X.shape}, classes: {np.bincount(y)}")

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True,
        random_state=RANDOM_STATE)

    fold_metrics = {v: {"ACC": [], "F1": [], "MCC": [],
                        "AUC": [], "Brier": []}
                    for v in VARIANTS}

    for fold, (train, test) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/{N_SPLITS}")
        X_train_raw, X_test = X[train], X[test]
        y_train_raw, y_test = y[train], y[test]

        # Fold-wise CTGAN
        X_train, y_train = ctgan_augment_minority(
            X_train_raw, y_train_raw,
            epochs=CTGAN_EPOCHS,
            random_state=RANDOM_STATE + fold)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        for name, cfg in VARIANTS.items():
            model = DendriticModel(
                input_dim=X.shape[1],
                n_branches=4,
                lr=0.042,
                epochs=493,
                seed=RANDOM_STATE + fold,
                **cfg)
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
            fold_metrics[name]["Brier"].append(
                brier_score_loss(y_test, prob))

    # =================================================
    # SUMMARY TABLE
    # =================================================
    print("\n" + "=" * 110)
    print("ABLATION STUDY — 10-FOLD CV  (mean ± std)")
    print("=" * 110)
    print(f"{'Variant':<20}{'ACC':>16}{'F1':>16}"
          f"{'MCC':>16}{'AUC':>18}{'Brier':>16}")
    print("-" * 110)

    summary = {}
    for v in VARIANT_ORDER:
        row = {}
        for k in ["ACC", "F1", "MCC", "AUC", "Brier"]:
            vals = np.array(fold_metrics[v][k])
            row[k] = (vals.mean(), vals.std())
        summary[v] = row
        print(f"{v:<20}"
              f"{row['ACC'][0]:>10.4f}±{row['ACC'][1]:<5.4f}"
              f"{row['F1'][0]:>10.4f}±{row['F1'][1]:<5.4f}"
              f"{row['MCC'][0]:>10.4f}±{row['MCC'][1]:<5.4f}"
              f"{row['AUC'][0]:>12.4f}±{row['AUC'][1]:<5.4f}"
              f"{row['Brier'][0]:>10.4f}±{row['Brier'][1]:<5.4f}")
    print("=" * 110)

    pd.DataFrame({
        v: {k: f"{summary[v][k][0]:.4f} ± {summary[v][k][1]:.4f}"
            for k in summary[v]}
        for v in VARIANT_ORDER
    }).T.to_csv(OUTPUT_DIR / "ablation_summary.csv")
    print(f"\nSaved: {OUTPUT_DIR / 'ablation_summary.csv'}")

    # =================================================
    # FRIEDMAN TEST
    # =================================================
    print("\n" + "=" * 75)
    print("FRIEDMAN TEST")
    print("=" * 75)
    for metric in ["AUC", "F1", "ACC", "MCC", "Brier"]:
        M = np.array([fold_metrics[v][metric]
                      for v in VARIANT_ORDER]).T
        stat, p = stats.friedmanchisquare(
            *[M[:, i] for i in range(M.shape[1])])
        sig = "***" if p < 0.001 else \
              "**" if p < 0.01 else \
              "*" if p < 0.05 else "ns"
        print(f"{metric:>6}:  χ² = {stat:8.4f}   "
              f"p = {p:.4e}   {sig}")

    # =================================================
    # WILCOXON (Full SDNM vs each variant)
    # =================================================
    print("\n" + "=" * 75)
    print("WILCOXON SIGNED-RANK (Full SDNM vs each variant)")
    print("=" * 75)

    REF = "Full SDNM+Adam"
    BASELINES = [v for v in VARIANT_ORDER if v != REF]

    for metric in ["AUC", "F1", "MCC", "ACC"]:
        ref = np.array(fold_metrics[REF][metric])
        raw_p, Ws = [], []
        for b in BASELINES:
            bv = np.array(fold_metrics[b][metric])
            try:
                W, p = stats.wilcoxon(
                    ref, bv, alternative="greater")
            except ValueError:
                W, p = 0.0, 1.0
            Ws.append(W); raw_p.append(p)
        adj_p = holm_bonferroni(raw_p)

        print(f"\n--- Metric: {metric} ---")
        print(f"{'Comparison':<35}{'W':>10}"
              f"{'p (raw)':>14}{'p (Holm)':>14}{'Sig':>6}")
        for b, w, p, ph in zip(BASELINES, Ws, raw_p, adj_p):
            sig = "***" if ph < 0.001 else \
                  "**" if ph < 0.01 else \
                  "*" if ph < 0.05 else "ns"
            print(f"{REF} vs {b:<20}"
                  f"{w:>10.1f}{p:>14.4e}{ph:>14.4e}{sig:>6}")

    # =================================================
    # NEMENYI POST-HOC
    # =================================================
    print("\n" + "=" * 75)
    print("NEMENYI POST-HOC (mean ranks)")
    print("=" * 75)
    for metric in ["AUC", "F1"]:
        mr, CD = nemenyi_posthoc(fold_metrics, metric)
        print(f"\n--- {metric} (CD = {CD:.4f}) ---")
        for idx in np.argsort(mr):
            print(f"  {VARIANT_ORDER[idx]:<20} "
                  f"rank = {mr[idx]:.3f}")

    # =================================================
    # FIGURE 1: BAR CHART
    # =================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
    palette = ["#7f7f7f", "#1f77b4", "#ff7f0e",
               "#2ca02c", "#d62728"]

    for ax, metric, label in zip(
        axes,
        ["ACC", "AUC"],
        ["Accuracy (ACC)", "Area Under the ROC Curve (AUC)"]
    ):
        means = [summary[v][metric][0] for v in VARIANT_ORDER]
        stds = [summary[v][metric][1] for v in VARIANT_ORDER]

        bars = ax.bar(VARIANT_ORDER, means, yerr=stds,
                      capsize=6, color=palette,
                      edgecolor="black", linewidth=1.1,
                      alpha=0.9,
                      error_kw={"elinewidth": 1.4,
                                "ecolor": "black"})
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2.0)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(means) * 0.02,
                    f"{mean:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_ylabel(label, fontsize=13, fontweight="bold")
        ax.set_title(f"Ablation — {metric}",
                     fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(means) * 1.18)

    plt.suptitle(
        "Ablation Study: Component Contributions to Performance",
        fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Ablation_BarChart_600dpi.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUT_DIR / "Ablation_BarChart_600dpi.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # =================================================
    # FIGURE 2: HEATMAP
    # =================================================
    metric_list = ["ACC", "F1", "MCC", "AUC"]
    mat = np.array([[summary[v][m][0] for m in metric_list]
                    for v in VARIANT_ORDER])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=600)
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(metric_list)))
    ax.set_xticklabels(metric_list, fontsize=12,
                       fontweight="bold")
    ax.set_yticks(np.arange(len(VARIANT_ORDER)))
    ax.set_yticklabels(VARIANT_ORDER, fontsize=11.5,
                       fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] < mat.max() * 0.9 \
                    else "black"
            ax.text(j, i, f"{mat[i, j]:.4f}",
                    ha="center", va="center",
                    color=color, fontsize=10.5,
                    fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(
        "Ablation Study — Mean Performance Across 10-Fold CV",
        fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Ablation_Heatmap_600dpi.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUT_DIR / "Ablation_Heatmap_600dpi.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # =================================================
    # FIGURE 3: CRITICAL DIFFERENCE DIAGRAMS
    # =================================================
    plot_critical_difference(
        fold_metrics, "AUC", "Ablation_CD_Diagram_AUC.png")
    plot_critical_difference(
        fold_metrics, "F1", "Ablation_CD_Diagram_F1.png")

    print("\nAll ablation outputs saved to:", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
