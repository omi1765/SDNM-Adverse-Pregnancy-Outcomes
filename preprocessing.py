import pandas as pd
import numpy as np


def load_and_clean_data(csv_path):
    """
    General data cleaning:
    - Fix column names
    - Harmonize hemoglobin variables
    """
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "Hypertnesion": "Hypertension",
        "Hb (g/L)": "Hb_g_L",
        "ST2(ng/ml)": "sST2",
        "NT-proBNP": "NT_proBNP",
        "CRP (mg/L)": "CRP",
        " Loop diuretics": "Loop_diuretics"
    })

    if "hemoglobin" in df.columns and "Hb_g_L" in df.columns:
        df["Hemoglobin"] = df["hemoglobin"]
        df["Hemoglobin"] = df["Hemoglobin"].fillna(df["Hb_g_L"])
        df = df.drop(columns=["hemoglobin", "Hb_g_L"])

    return df


def load_imbalanced_data(csv_path, target):
    """
    Dataset preparation for imbalanced learning:
    - Remove identifiers
    - Remove perfect separation variables
    - Numeric coercion and median imputation
    """
    df = load_and_clean_data(csv_path)

    drop_basic = ["Admission number ", "Date of admission"]
    df = df.drop(columns=[c for c in drop_basic if c in df.columns])

    perfect_sep_vars = [
        'LVEF','low_birthweight','sga','preterm',
        'spontaneous_preterm','stillbirth',
        'pregnancies','glucose','blood_pressure',
        'skinthickness','insulin','age'
    ]
    df = df.drop(columns=[c for c in perfect_sep_vars if c in df.columns])

    y = df[target]
    X = df.drop(columns=[target])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, y
