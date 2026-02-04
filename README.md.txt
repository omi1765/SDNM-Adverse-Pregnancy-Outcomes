Multi-Layer Hybrid Modeling of Adverse Pregnancy Outcomes Using a Stable Dendritic Neural Model (SDNM)


## Overview
This repository presents a multi-layer analytical framework for studying
adverse pregnancy outcomes (APOs) in women with pre-existing cardiac disease.
The work integrates observational analysis, synthetic data generation, and
advanced machine learning to investigate the predictive and mechanistic role
of soluble ST2 (sST2), a biomarker of cardiac stress and fibrosis.

A novel Stable Dendritic Neural Model (SDNM) is proposed to improve predictive
stability, interpretability, and performance in small and imbalanced
clinical datasets.

---

## Clinical Motivation
Adverse pregnancy outcomes (APOs), including preeclampsia, preterm birth,
stillbirth, and heart failure, pose significant risks for women with cardiac
disease. Although sST2 is a well-established biomarker in cardiology, its
predictive utility in pregnancy-related cardiac risk remains underexplored.

A major barrier is the lack of large-scale cohorts with jointly measured
sST2 and pregnancy outcomes. This study addresses this limitation using
multi-layer data integration and machine learning–based risk modeling.

---

## Methodological Framework

### Stage 1: Observational Association
- Multivariable regression and stratified analysis
- Adjustment for confounders:
  - Age
  - BMI
  - Hypertension
  - Diabetes
  - Lifestyle factors
- Association between sST2 and composite APO outcomes

### Stage 2: Predictive Modeling and Mechanistic Insight
- Synthetic data generation using CTGAN to address class imbalance
- Comparative modeling using:
  - Decision Tree (DT)
  - Random Forest (RF)
  - Gradient Boosting (GBM)
  - XGBoost (XGB)
  - LightGBM (LGBM)
- Proposed Stable Dendritic Neural Model (SDNM)
- Evaluation via 10-fold stratified cross-validation
- Metrics:
  - AUC
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - F1-score
  - Cohen’s Kappa
  - Matthews Correlation Coefficient (MCC)

---

## Stable Dendritic Neural Model (SDNM)
The SDNM is a biologically inspired neural architecture designed to capture
nonlinear, multiplicative feature interactions while maintaining numerical
stability. Key characteristics include:

- Dendritic branch-wise multiplicative integration
- Log-domain stabilization for numerical robustness
- Adam optimization for improved convergence
- Enhanced performance in imbalanced and limited-sample clinical data

## Model Interpretability: SHAP and LIME

To ensure clinical transparency and trustworthiness, post-hoc model
interpretability techniques were incorporated into the analytical pipeline.

### SHAP (SHapley Additive exPlanations)
SHAP was used to quantify global and local feature contributions across
tree-based models and the proposed SDNM. Mean absolute SHAP values were
computed to identify dominant clinical predictors influencing APO risk,
including sST2, BMI, age, and comorbidity indicators.

SHAP summary plots were employed to visualize feature importance and
directionality, enabling clinically meaningful interpretation of nonlinear
feature effects.

### LIME (Local Interpretable Model-Agnostic Explanations)
LIME was applied to generate instance-level explanations for individual
predictions. This allowed localized understanding of risk attribution for
high-risk pregnancies, supporting case-based clinical reasoning.

Together, SHAP and LIME enhance the explainability, transparency, and
regulatory alignment of the proposed machine learning framework.

├── preprocessing.py        # Data cleaning, feature engineering, imputation
├── ctgan.py                # Synthetic data generation for class balancing
├── imbalance_base_dnm.py   # Base models and DNM on imbalanced data
├── balance_base_dnm.py     # Base models and DNM on balanced data
├── stable_dnm.py           # Proposed Stable Dendritic Neural Model (SDNM)
├── explainability/
│   ├── shap_analysis.py    # SHAP-based global and local explanations
│   └── lime_analysis.py    # LIME-based instance explanations
└── README.md
