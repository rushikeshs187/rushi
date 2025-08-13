# modeling.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def time_split(df: pd.DataFrame, train_size=0.6, val_size=0.2):
    n = len(df)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test

def _prepare_xy(df: pd.DataFrame):
    y = df["Target"].astype(int).values
    X = df.drop(columns=["Target"]).select_dtypes(include=[np.number]).values
    idx = df.index
    return X, y, idx

def _scale(train_X, val_X, test_X):
    sc = StandardScaler()
    return sc.fit_transform(train_X), sc.transform(val_X), sc.transform(test_X)

def _models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "SVM-RBF": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             alpha=1e-4, learning_rate_init=1e-3, max_iter=400,
                             random_state=42, shuffle=False, early_stopping=True,
                             n_iter_no_change=20, validation_fraction=0.15)
    }

def train_and_score_models(ml_df: pd.DataFrame):
    # Keep numeric columns only
    work = ml_df.select_dtypes(include=[np.number]).copy()
    # Split
    train_df, val_df, test_df = time_split(work, 0.6, 0.2)
    # X/y
    X_tr, y_tr, idx_tr = _prepare_xy(train_df)
    X_va, y_va, idx_va = _prepare_xy(val_df)
    X_te, y_te, idx_te = _prepare_xy(test_df)
    # Scale
    X_tr, X_va, X_te = _scale(X_tr, X_va, X_te)

    res = []
    preds = {}
    for name, clf in _models().items():
        clf.fit(X_tr, y_tr)
        # Validation
        v_pred = clf.predict(X_va)
        v_proba = clf.predict_proba(X_va)[:, 1]
        v_acc = accuracy_score(y_va, v_pred)
        v_f1 = f1_score(y_va, v_pred, zero_division=0)
        v_auc = roc_auc_score(y_va, v_proba)

        # Test
        t_pred = clf.predict(X_te)
        t_proba = clf.predict_proba(X_te)[:, 1]
        t_acc = accuracy_score(y_te, t_pred)
        t_f1 = f1_score(y_te, t_pred, zero_division=0)
        t_auc = roc_auc_score(y_te, t_proba)

        res.append({
            "Model": name,
            "Val_Acc": v_acc, "Val_F1": v_f1, "Val_AUC": v_auc,
            "Test_Acc": t_acc, "Test_F1": t_f1, "Test_AUC": t_auc,
        })
        preds[name] = {
            "test_proba": pd.Series(t_proba, index=idx_te),
            "price_test": None,   # injected by caller
            "index_test": idx_te
        }

    # inject price for backtest
    # reconstruct series from original frame indices if possible
    price_col = "Adj Close" if "Adj Close" in ml_df.columns else "Close"
    price_series = ml_df[price_col].loc[idx_te]
    for name in preds:
        preds[name]["price_test"] = price_series

    results = pd.DataFrame(res).sort_values("Val_AUC", ascending=False)
    return results, preds

def backtest_from_proba(price: pd.Series, proba: pd.Series, index: pd.DatetimeIndex,
                        prob_threshold=0.5, cost_bps=5, periods=252):
    # Align
    proba = pd.Series(proba, index=index).sort_index()
    price = pd.Series(price, index=index).sort_index()
    # Positions (enter next day to avoid look-ahead)
    pos = (proba >= prob_threshold).astype(int)
    ret = price.pct_change().fillna(0.0)
    strat = ret * pos.shift(1).fillna(0)

    # Apply simple transaction costs on position changes
    trades = pos.diff().abs().fillna(0)
    tc = trades * (cost_bps / 10000.0)  # bps to decimal
    strat_after_cost = strat - tc

    equity = (1 + strat_after_cost).cumprod()
    # Metrics
    if equity.empty:
        return {"equity": equity, "CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan}

    n_years = len(equity) / periods
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = (strat_after_cost.mean() / strat_after_cost.std(ddof=0)) * np.sqrt(periods) if strat_after_cost.std(ddof=0) > 0 else np.nan
    downside = strat_after_cost[strat_after_cost < 0]
    sortino = (strat_after_cost.mean() / downside.std(ddof=0)) * np.sqrt(periods) if downside.std(ddof=0) > 0 else np.nan
    peak = equity.cummax()
    mdd = (equity / peak - 1.0).min()

    return {"equity": equity, "CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": mdd}
