# ctgan.py

import pandas as pd
import numpy as np
from ctgan import CTGAN


def ctgan_balance(
    X,
    y,
    target_name,
    epochs=300,
    random_state=42
):
    """
    Balance an imbalanced dataset using CTGAN.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target labels
    target_name : str
        Name of target column
    epochs : int
        Number of CTGAN training epochs
    random_state : int
        Random seed

    Returns
    -------
    X_balanced : pd.DataFrame
        Balanced feature matrix
    y_balanced : pd.Series
        Balanced target vector
    """


    if isinstance(y, np.ndarray):
        y = pd.Series(y, name=target_name)

    # Combine X and y
    df = pd.concat(
        [X.reset_index(drop=True),
         y.reset_index(drop=True)],
        axis=1
    )

   
    class_counts = df[target_name].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    n_to_generate = class_counts[majority_class] - class_counts[minority_class]

    # Already balanced
    if n_to_generate <= 0:
        return X.copy(), y.copy()

    # Separate minority class
    df_minority = df[df[target_name] == minority_class]

    # CTGAN requires numeric & no missing values
    df_minority = df_minority.apply(pd.to_numeric, errors="coerce")
    df_minority = df_minority.replace([np.inf, -np.inf], np.nan)
    df_minority = df_minority.fillna(df_minority.median())

    # Train CTGAN
    ctgan = CTGAN(
        epochs=epochs,
        random_state=random_state,
        verbose=False
    )
    ctgan.fit(df_minority, discrete_columns=[target_name])

    # Generate synthetic minority samples
    synthetic = ctgan.sample(n_to_generate)
    synthetic = synthetic[synthetic[target_name] == minority_class]

    # Combine original + synthetic
    df_balanced = pd.concat([df, synthetic], ignore_index=True)

    # Split back to X, y
    X_balanced = df_balanced.drop(columns=[target_name])
    y_balanced = df_balanced[target_name]

    return X_balanced, y_balanced
