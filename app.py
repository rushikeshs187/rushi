# IBR Finance — Fully Dark Research Dashboard
# EDA → Features → Modeling → Backtest & Risk → Insights
# Polished dark reset (no white flashes), crisp KPIs, consistent Plotly styling.

# --- Safety net: install locally if needed (Cloud uses requirements.txt) ---
def _ensure(pkgs):
    import importlib, sys, subprocess
    for name, pip_name in pkgs:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

_ensure([
    ("streamlit","streamlit>=1.38"),
    ("pandas","pandas>=1.5"),
    ("numpy","numpy>=1.23"),
    ("yfinance","yfinance>=0.2.40"),
    ("plotly","plotly>=5.24"),
    ("sklearn","scikit-learn>=1.3"),
])

# --- Imports ---
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# --- Page config ---
st.set_page_config(
    page_title="IBR Finance — Research Dashboard",
    page_icon="🌓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DARK RESET (no white anywhere) ---
st.markdown("""
<style>
:root{
  color-scheme: dark;
  --bg:#0b0e13;            /* page */
  --panel:#10151d;         /* sidebar/header/panels */
  --card:#131a24;          /* cards/tables */
  --border:#223044;        /* strokes */
  --text:#e6edf3;          /* main text */
  --muted:#95a3b3;         /* secondary text */
  --accent:#5f8cff;        /* brand accent */
  --accent-2:#ff6b6b;      /* danger */
  --radius:16px;
}
/* page, header, sidebar */
.stApp, html, body, .block-container{ background:var(--bg)!important; }
[data-testid="stHeader"], [data-testid="stToolbar"]{ background:var(--panel)!important; border-bottom:1px solid var(--border); }
[data-testid="stSidebar"]{ background:var(--panel)!important; border-right:1px solid var(--border); }
.block-container{ padding-top:0; padding-bottom:32px }

/* sticky header */
.header{ position:sticky; top:0; z-index:50; background:linear-gradient(180deg,var(--panel),rgba(16,21,29,.75));
  backdrop-filter:blur(6px); border-bottom:1px solid var(--border); padding:16px 12px 10px; margin:0 -12px 18px }
.header h1{ margin:0; font-size:1.65rem; color:var(--text) }
.header .sub{ color:var(--muted); font-size:.95rem; margin-top:2px }

/* toolbar badges */
.toolbar{ display:flex; gap:12px; align-items:center; border:1px solid var(--border); border-radius:var(--radius);
  background:var(--panel); padding:8px 12px; margin-top:10px }
.badge{ background:#0f1722; border:1px solid var(--border); color:var(--muted); padding:5px 10px; border-radius:999px; font-size:.85rem }

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:18px; border-bottom:1px solid var(--border) }
.stTabs [data-baseweb="tab"]{ color:var(--text) }
.stTabs [aria-selected="true"]{ border-bottom:3px solid var(--accent)!important }

/* cards / kpis */
.card, .kpi{ background:var(--card)!important; border:1px solid var(--border)!important; border-radius:var(--radius); color:var(--text) }
.kpi{ padding:18px 20px; height:118px; display:flex; flex-direction:column; justify-content:center; box-shadow:0 1px 0 rgba(0,0,0,.25) }
.kpi .label{ color:var(--muted)!important; font-size:.95rem; margin-bottom:5px }
.kpi .value{ color:var(--text)!important; font-size:1.9rem; font-weight:800 }

/* text & hr */
h1,h2,h3,h4,h5,h6,.stMarkdown,.stMarkdown p,label{ color:var(--text)!important; opacity:1!important }
hr.hr{ border:0; height:1px; background:var(--border); margin:12px 0 16px }

/* buttons & downloads */
.stButton>button, .stDownloadButton>button{
  background:var(--panel)!important; color:var(--text)!important; border:1px solid var(--border)!important; border-radius:10px
}
.stButton>button:hover, .stDownloadButton>button:hover{ background:#0f1722!important }

/* inputs */
[data-baseweb="select"]>div, .stTextInput>div>div, .stNumberInput>div>div, .stDateInput>div>div{
  background:var(--panel)!important; color:var(--text)!important; border:1px solid var(--border)!important;
}
[data-baseweb="select"] svg{ fill:var(--muted)!important }

/* slider */
.stSlider [role="slider"]{ background:var(--accent)!important }

/* alerts */
.stAlert{ background:var(--panel)!important; border:1px solid var(--border)!important; color:var(--text)!important }

/* dataframe (headers, rows, footer) */
[data-testid="stDataFrame"] *{ color:var(--text)!important }
[data-testid="stDataFrame"] div, [data-testid="stDataFrame"] canvas{ background:var(--card)!important }
[data-testid="stDataFrame"] [role="table"]{ background:var(--card)!important }

/* code blocks */
pre, code{ background:var(--panel)!important; color:var(--text)!important; border:1px solid var(--border) }

/* scrollbars */
*::-webkit-scrollbar{ height:10px; width:10px }
*::-webkit-scrollbar-track{ background:var(--panel) }
*::-webkit-scrollbar-thumb{ background:#1d2633; border:2px solid var(--panel); border-radius:10px }
</style>
""", unsafe_allow_html=True)

# --- Plotly theme (dark tooltips/legend too) ---
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0b0e13",
    plot_bgcolor="#0b0e13",
    font_color="#e6edf3",
    hoverlabel_bgcolor="#10151d",
    hoverlabel_bordercolor="#223044",
    hoverlabel_font_color="#e6edf3",
    legend_bgcolor="#0b0e13",
)

# --- Universes / Presets ---
UNIVERSES = {
    "US (Developed)": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","JPM","XOM"],
    "UK (Developed)": ["HSBA.L","AZN.L","BP.L","ULVR.L","GSK.L","RIO.L"],
    "India (Emerging)": ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"],
    "Brazil (Emerging)": ["VALE3.SA","PETR4.SA","ITUB4.SA","B3SA3.SA"],
    "South Africa (Emerging)": ["NPN.JO","AGL.JO","BHG.JO","SOL.JO"],
    "Crypto (Global)": ["BTC-USD","ETH-USD"],
}
INTERVALS = ["1d","1wk","1mo"]
PERIODS = ["1y","2y","5y","10y","max"]
PRESETS = {
    "Developed study": ["US (Developed)", ["AAPL","MSFT","NVDA","AMZN","GOOGL","META"]],
    "Emerging study":  ["India (Emerging)", ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]],
}

# --- Helpers ---
def compute_indicators(df):
    def rsi(x, n=14):
        d = x.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        ag = g.rolling(n).mean(); al = l.rolling(n).mean().replace(0, np.nan)
        rs = ag/al; return 100 - (100/(1+rs))
    out=[]
    for _, g in df.groupby("Symbol", sort=False):
        g=g.sort_values("Date").copy()
        px=g["Adj Close"].astype(float)
        g["Return"]=px.pct_change(); g["LogRet"]=np.log(px).diff()
        g["SMA_20"]=px.rolling(20).mean(); g["SMA_50"]=px.rolling(50).mean()
        g["EMA_12"]=px.ewm(span=12, adjust=False).mean(); g["EMA_26"]=px.ewm(span=26, adjust=False).mean()
        g["MACD"]=g["EMA_12"]-g["EMA_26"]; g["MACD_Signal"]=g["MACD"].ewm(span=9, adjust=False).mean()
        g["BB_M"]=px.rolling(20).mean(); s=px.rolling(20).std()
        g["BB_U"]=g["BB_M"]+2*s; g["BB_L"]=g["BB_M"]-2*s
        g["RSI_14"]=rsi(px,14); g["Volatility_20"]=g["Return"].rolling(20).std()*np.sqrt(252)
        out.append(g)
    return pd.concat(out, axis=0) if out else pd.DataFrame()

def prepare_ml_frame(df, horizon=1):
    feats=["Return","LogRet","SMA_20","SMA_50","EMA_12","EMA_26","MACD","MACD_Signal","RSI_14","BB_M","BB_U","BB_L","Volatility_20","LagRet_1","LagRet_2","LagRet_5"]
    if df.empty: return pd.DataFrame(), feats
    frames=[]
    for _, g in df.groupby("Symbol", sort=False):
        g=g.sort_values("Date").copy()
        g["LagRet_1"]=g["LogRet"].shift(1); g["LagRet_2"]=g["LogRet"].shift(2); g["LagRet_5"]=g["LogRet"].shift(5)
        g["Target"]=(g["LogRet"].shift(-horizon)>0).astype(int)
        frames.append(g)
    X=pd.concat(frames, axis=0) if frames else pd.DataFrame()
    if X.empty: return X, feats
    X=X.dropna(subset=feats+["Target"]).copy(); return X, feats

def sharpe_ratio(r, rf=0.0, ppy=252):
    s=pd.Series(r).dropna()
    if s.empty: return np.nan
    mu=(s - rf/ppy).mean()*ppy; sig=s.std(ddof=0)*np.sqrt(ppy)
    return np.nan if sig==0 or np.isnan(sig) else mu/sig

def max_drawdown(c):
    s=pd.Series(c).replace([np.inf,-np.inf], np.nan).ffill()
    if s.empty: return np.nan
    m=s.cummax(); dd=s/m - 1.0; return float(dd.min())

@st.cache_data(show_spinner=True)
def fetch_prices(symbols, period="5y", interval="1d"):
    if isinstance(symbols,str): symbols=[symbols]
    data=yf.download(
        tickers=" ".join(symbols), period=period, interval=interval,
        auto_adjust=False, progress=False, threads=True, group_by="ticker"
    )
    if data is None or len(data)==0:
        return pd.DataFrame(columns=["Symbol","Date","Open","High","Low","Close","Adj Close","Volume"])
    frames=[]
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                g=data[sym].reset_index()
                if "Date" not in g.columns: g.rename(columns={"index":"Date"}, inplace=True)
                g.insert(0,"Symbol",sym); frames.append(g)
    else:
        g=data.reset_index()
        if "Date" not in g.columns: g.rename(columns={"index":"Date"}, inplace=True)
        g.insert(0,"Symbol",symbols[0]); frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["Symbol","Date","Open","High","Low","Close","Adj Close","Volume"])
    df=pd.concat(frames, axis=0, ignore_index=True)
    df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c not in df.columns: df[c]=np.nan
        df[c]=pd.to_numeric(df[c], errors="coerce")
    df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
    df=df.dropna(subset=["Date"]).sort_values(["Symbol","Date"])
    df=df[~df[["Date","Symbol"]].duplicated(keep="last")]
    return df.reset_index(drop=True)

def dl_button(df, label, name):
    if df is not None and not df.empty:
        st.download_button(label, df.to_csv(index=False).encode("utf-8"), name, "text/csv")

def kpi(col, label, value):
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------- Sidebar ---------------------------
st.sidebar.markdown("### ⚙️ Study Controls")
preset = st.sidebar.selectbox("Preset", ["— None —"] + list(PRESETS.keys()))
if preset!="— None —":
    p_market, p_symbols = PRESETS[preset]
else:
    p_market, p_symbols = list(UNIVERSES.keys())[0], UNIVERSES[list(UNIVERSES.keys())[0]]

market = st.sidebar.selectbox("Market", list(UNIVERSES.keys()), index=list(UNIVERSES.keys()).index(p_market))
symbols_default = UNIVERSES[market]
symbols = st.sidebar.multiselect("Symbols", options=symbols_default, default=(p_symbols if preset!="— None —" else symbols_default))
period = st.sidebar.selectbox("History Window", PERIODS, index=2)
interval = st.sidebar.selectbox("Sampling Interval", INTERVALS, index=0)
eda_symbol = st.sidebar.selectbox("Primary symbol (deep-dive)", options=symbols if symbols else symbols_default, index=0)

with st.sidebar.expander("Advanced Modeling", expanded=False):
    horizon = st.slider("Prediction horizon (days ahead)", 1, 5, 1, 1)
    test_size_years = st.slider("Test span (years)", 1, 5, 2, 1)
    n_splits = st.slider("CV splits (rolling)", 2, 5, 3, 1)
    rf_trees = st.slider("RF: n_estimators", 50, 400, 200, 50)
    svm_c = st.selectbox("SVM: C", [0.1,0.5,1.0,2.0,5.0], index=2)
    mlp_hidden = st.selectbox("ANN (MLP): hidden units", [32,64,128], index=1)

# --------------------------- Header ---------------------------
st.markdown(f"""
<div class="header">
  <h1>IBR Finance — Markets EDA & Model Comparison</h1>
  <div class="sub">Exploration → Features → Models → Backtest → Insights</div>
  <div class="toolbar">
    <span class="badge">🌍 {market}</span>
    <span class="badge">📦 {len(symbols)} symbols</span>
    <span class="badge">🗓 {period}</span>
    <span class="badge">⏱ {interval}</span>
    <span class="badge">🎯 horizon {horizon}d</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------- Data ---------------------------
if not symbols:
    st.warning("Pick at least one symbol to proceed."); st.stop()

with st.spinner("Fetching market data..."):
    raw = fetch_prices(symbols, period=period, interval=interval)
if raw.empty:
    st.error("No data returned for selected inputs. Try another market/period/interval."); st.stop()

raw = raw.sort_values(["Symbol","Date"]).reset_index(drop=True)
enriched = compute_indicators(raw) if not raw.empty else pd.DataFrame()

# --------------------------- Tabs ---------------------------
tab_overview, tab_eda, tab_feat, tab_model, tab_bt, tab_insights = st.tabs(
    ["Overview", "EDA", "Features", "Modeling", "Backtest & Risk", "Insights"]
)

# ===== Overview =====
with tab_overview:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    kpi(c1, "Symbols", f"{raw['Symbol'].nunique():,}")
    kpi(c2, "Observations", f"{len(raw):,}")
    kpi(c3, "Date Range", f"{raw['Date'].min().date()} → {raw['Date'].max().date()}")
    kpi(c4, "Interval", f"{interval}")
    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    sym = eda_symbol
    sym_df = raw[raw["Symbol"]==sym].dropna(subset=["Adj Close"]).copy()
    sym_df["SMA_20"] = sym_df["Adj Close"].rolling(20).mean()
    sym_df["SMA_50"] = sym_df["Adj Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["Adj Close"], mode="lines", name=f"{sym} Adj Close"))
    fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["SMA_20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["SMA_50"], mode="lines", name="SMA 50"))
    fig.update_layout(height=380, title=f"{sym} — Price & SMAs", xaxis_title=None, yaxis_title="Adj Close", **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([3,1])
    with left:
        st.markdown('<div class="card">Download datasets</div>', unsafe_allow_html=True)
        dl_button(raw, "⬇️ Prices (CSV)", f"prices_{market.replace(' ','_')}.csv")
        if not enriched.empty:
            dl_button(enriched, "⬇️ Enriched features (CSV)", "enriched_features.csv")
    with right:
        st.empty()

# ===== EDA =====
with tab_eda:
    sym_df = raw[raw["Symbol"]==eda_symbol].dropna(subset=["Adj Close"]).copy()
    sym_df["Return"]=sym_df["Adj Close"].pct_change()
    sym_df["LogRet"]=np.log(sym_df["Adj Close"]).diff()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        hist = px.histogram(sym_df, x="Return", nbins=60, title=f"{eda_symbol} — Daily Return Distribution")
        hist.update_layout(height=320, **PLOTLY_THEME)
        st.plotly_chart(hist, use_container_width=True)
    with col2:
        roll = sym_df.set_index("Date")["Return"].rolling(20).std()*np.sqrt(252)
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Ann. Vol (20d)"))
        rfig.update_layout(height=320, title=f"{eda_symbol} — Rolling Annualized Volatility (20d)", **PLOTLY_THEME)
        st.plotly_chart(rfig, use_container_width=True)

    with st.expander("Summary stats"):
        st.dataframe(sym_df[["Adj Close","Return","LogRet","Volume"]].describe().T, use_container_width=True)

# ===== Features =====
with tab_feat:
    if enriched.empty:
        st.info("Insufficient data to compute indicators.")
    else:
        latest = (
            enriched.sort_values("Date")
            .groupby("Symbol", as_index=False)
            .apply(lambda g: g.iloc[-1][["Symbol","Date","Adj Close","RSI_14","Volatility_20"]])
            .reset_index(drop=True)
        )
        st.markdown("**Latest snapshot**")
        st.dataframe(latest.style.format({"Adj Close":"{:,.2f}", "RSI_14":"{:,.1f}", "Volatility_20":"{:,.2f}"}),
                     use_container_width=True)

        corr_cols=["Return","LogRet","SMA_20","SMA_50","EMA_12","EMA_26","MACD","MACD_Signal","RSI_14","Volatility_20"]
        corr_df=enriched[enriched["Symbol"]==eda_symbol][corr_cols].dropna()
        if len(corr_df)>30:
            cfig = px.imshow(corr_df.corr(numeric_only=True), text_auto=True, aspect="auto", title=f"{eda_symbol} — Feature Correlation")
            cfig.update_layout(height=520, **PLOTLY_THEME)
            st.plotly_chart(cfig, use_container_width=True)
        else:
            st.info("Not enough rows (>30) for a stable correlation heatmap.")

# ===== Modeling =====
with tab_model:
    ml_df, features = prepare_ml_frame(enriched, horizon=horizon) if not enriched.empty else (pd.DataFrame(), [])
    if ml_df.empty:
        st.warning("Not enough data after feature engineering. Try a longer period or daily interval."); st.stop()

    max_date=ml_df["Date"].max(); cutoff=max_date - pd.DateOffset(years=test_size_years)
    train=ml_df[ml_df["Date"]<cutoff].copy(); test=ml_df[ml_df["Date"]>=cutoff].copy()
    if train["Target"].nunique()<2 or test["Target"].nunique()<2:
        st.warning("Target class imbalance or insufficient variety. Increase history window."); st.stop()

    X_train, y_train = train[features].values, train["Target"].values
    X_test,  y_test  = test[features].values,  test["Target"].values

    models = {
        "RandomForest": Pipeline([("rf", RandomForestClassifier(n_estimators=rf_trees, random_state=42, n_jobs=-1))]),
        "SVM (RBF)"  : Pipeline([("sc", StandardScaler()), ("svm", SVC(C=svm_c, kernel="rbf", gamma="scale", probability=True, random_state=42))]),
        "ANN (MLP)"  : Pipeline([("sc", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(mlp_hidden,), activation="relu", solver="adam", max_iter=300, random_state=42))]),
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)

    with st.spinner("Training models..."):
        rows=[]; proba_dict={}; preds_dict={}
        for name, pipe in models.items():
            cv=[]
            for tr, va in tscv.split(X_train):
                p=pipe.fit(X_train[tr], y_train[tr])
                cv.append(accuracy_score(y_train[va], p.predict(X_train[va])))
            pipe.fit(X_train, y_train)
            y_hat=pipe.predict(X_test)
            if hasattr(pipe[-1],"predict_proba"): y_proba=pipe.predict_proba(X_test)[:,1]
            elif hasattr(pipe[-1],"decision_function"): d=pipe.decision_function(X_test); y_proba=1/(1+np.exp(-d))
            else: y_proba=y_hat.astype(float)
            proba_dict[name]=y_proba; preds_dict[name]=y_hat
            rows.append({
                "Model":name, "CV Acc (mean)":float(np.mean(cv)),
                "Test Acc":accuracy_score(y_test,y_hat),
                "Precision":precision_score(y_test,y_hat,zero_division=0),
                "Recall":recall_score(y_test,y_hat,zero_division=0),
                "F1":f1_score(y_test,y_hat,zero_division=0),
                "ROC-AUC":roc_auc_score(y_test,y_proba) if len(np.unique(y_test))==2 else np.nan
            })
        metrics=pd.DataFrame(rows).sort_values("Test Acc", ascending=False)

    st.subheader("Out-of-sample metrics")
    st.dataframe(metrics.style.format({"CV Acc (mean)":"{:.3f}","Test Acc":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}","ROC-AUC":"{:.3f}"}), use_container_width=True)
    dl_button(metrics, "⬇️ Metrics (CSV)", "model_metrics.csv")

    st.subheader("Confusion matrices")
    cols = st.columns(3, gap="large")
    for (name, yhat), c in zip(list(preds_dict.items())[:3], cols):
        cm=confusion_matrix(y_test, yhat)
        z=pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
        fig_cm=px.imshow(z, text_auto=True, aspect="auto", title=name)
        fig_cm.update_layout(height=300, margin=dict(l=20,r=20,t=60,b=20), **PLOTLY_THEME)
        c.plotly_chart(fig_cm, use_container_width=True)

    # stash for backtest
    st.session_state["test_df"]=test[["Date","Symbol","LogRet"]].reset_index(drop=True)
    st.session_state["probas"]=proba_dict

# ===== Backtest & Risk =====
with tab_bt:
    bt=st.session_state.get("test_df"); probas=st.session_state.get("probas")
    if bt is None or probas is None:
        st.info("Run the Modeling tab first.")
    else:
        curves={}; risk_rows=[]
        for name, proba in probas.items():
            if len(proba)!=len(bt): proba=np.resize(proba, len(bt))
            sig=(proba>0.5).astype(int).astype(float)
            pos=pd.Series(sig).shift(1).fillna(0.0).values
            df_tmp=bt.copy(); df_tmp["StratRet"]=pos*df_tmp["LogRet"].values
            eq=df_tmp.groupby("Date")["StratRet"].mean(); cum=(1+eq).cumprod()
            curves[name]=cum; risk_rows.append({"Model":name,"Sharpe":sharpe_ratio(eq.values),"Max Drawdown":max_drawdown(cum.values)})

        fig_bt=go.Figure()
        for name, curve in curves.items():
            fig_bt.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=name))
        fig_bt.update_layout(height=380, title="Equity Curves — equal-weight, long/flat by model signal",
                             yaxis_title="Growth of $1", **PLOTLY_THEME)
        st.plotly_chart(fig_bt, use_container_width=True)

        risk=pd.DataFrame(risk_rows).sort_values("Sharpe", ascending=False)
        st.dataframe(risk.style.format({"Sharpe":"{:.2f}","Max Drawdown":"{:.1%}"}), use_container_width=True)
        dl_button(pd.concat([v.rename(k) for k,v in curves.items()], axis=1).reset_index().rename(columns={"index":"Date"}),
                  "⬇️ Equity curves (CSV)", "equity_curves.csv")
        dl_button(risk, "⬇️ Risk metrics (CSV)", "risk_metrics.csv")

# ===== Insights =====
with tab_insights:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Insights & Alignment")
    st.markdown("""
- **Flow**: EDA → Feature engineering → Modeling → Backtest mirrors a research workflow.
- **Comparability**: RF, SVM, MLP trained on identical features with rolling CV.
- **Interpretability**: correlations (and optionally RF importances) justify signal.
- **Risk**: backtest outputs **Sharpe** and **Max Drawdown** for decisions.
- **Extend**: add macro factors, sector tags, slippage/costs, thresholding, and hyper-tuning.
""")
    st.markdown('</div>', unsafe_allow_html=True)
