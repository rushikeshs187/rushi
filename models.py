from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5, min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "SVM_RBF": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
    "ANN_MLP": MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                             solver="adam", learning_rate_init=1e-3,
                             max_iter=200, random_state=42)
}

def _feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"Symbol","Open","High","Low","Close","Price","TargetNextUp"}
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "fc"]

def time_split_train_eval(df: pd.DataFrame, n_splits: int = 4) -> Dict[str, Dict]:
    """TimeSeries CV on pooled panel; symbols mixed but time-ordered within each symbol."""
    feats = _feature_cols(df)
    X = df[feats].values
    y = df["TargetNextUp"].values
    tss = TimeSeriesSplit(n_splits=n_splits)
    results: Dict[str, Dict] = {name: {"acc": [], "auc": []} for name in CLASSIFIERS}

    for train_idx, test_idx in tss.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        for name, clf in CLASSIFIERS.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(Xtr, ytr)
            prob = pipe.predict_proba(Xte)[:,1]
            pred = (prob >= 0.5).astype(int)
            acc = accuracy_score(yte, pred)
            try:
                auc = roc_auc_score(yte, prob)
            except Exception:
                auc = np.nan
            results[name]["acc"].append(acc)
            results[name]["auc"].append(auc)
    # aggregate
    for name in results:
        for k in ("acc","auc"):
            arr = np.array(results[name][k], float)
            results[name][k] = float(np.nanmean(arr))
    return results

def fit_best_and_signal(df: pd.DataFrame, best_model_name: str) -> Tuple[Pipeline, pd.Series]:
    feats = _feature_cols(df)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", CLASSIFIERS[best_model_name])])
    # drop last row (no label for last next-day)
    df_train = df.iloc[:-1].dropna(subset=feats+["TargetNextUp"]).copy()
    pipe.fit(df_train[feats].values, df_train["TargetNextUp"].values)
    proba = pipe.predict_proba(df[feats].values)[:,1]
    signal = pd.Series((proba >= 0.5).astype(int), index=df.index, name="Signal")
    return pipe, signal
