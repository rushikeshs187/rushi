import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML in Finance – Research Dashboard", layout="wide")

# -------- Market & File Setup ----------
markets = {
    'S&P 500': 'sp500_full_features.csv',
    'Nifty 50': 'nifty50_full_features.csv',
    'FTSE 100': 'ftse100_full_features.csv',
    'Bovespa': 'bovespa_full_features.csv'
}

# --- Sidebar: Market and Ticker Filters ---
st.sidebar.header("Dashboard Controls")
market = st.sidebar.selectbox("Select Market", list(markets.keys()), help="Choose which market to analyze.")
df = pd.read_csv(markets[market])
ticker = st.sidebar.selectbox("Select Ticker", sorted(df['Ticker'].unique()), help="Choose company to visualize.")

sample = df[df['Ticker'] == ticker]

st.title("Machine Learning in Finance: Multi-Market Research Dashboard")
st.markdown("""
**This dashboard is built to support research objectives:**
- *Assess real-world impact of ML models on investment decisions*
- *Benchmark and compare Random Forest, SVM, ANN *
- *Compare developed vs. emerging markets *
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Data Visualization",
    "2. ML Model Results ",
    "3. Interpretation ",
    "4. About & Research Objectives"
])

# -------- TAB 1: Data Visualization --------
with tab1:
    st.header(f"EDA for {ticker} in {market}")
    st.write("**A. Price Trend** – Visualizes long-term movement, key for asset selection, regime shifts, and real-world investment insight.")

    fig, ax = plt.subplots()
    ax.plot(sample['Date'], sample['Close'], color='navy')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.set_title(f"{ticker} Closing Price")
    st.pyplot(fig)

    st.write("**B. Daily Return Distribution** – Shows market risk/volatility; 'fatter' tails mean higher risk of extreme moves.")

    fig2, ax2 = plt.subplots()
    ax2.hist(sample['Return'].dropna(), bins=50, color='teal')
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    st.write("**C. 20-Day SMA vs Close** – Moving average (trend-following) helps visualize smoothing, momentum for strategies like asset allocation.")

    fig3, ax3 = plt.subplots()
    ax3.plot(sample['Date'], sample['Close'], label='Close', color='navy')
    if 'SMA_20' in sample.columns:
        ax3.plot(sample['Date'], sample['SMA_20'], label='SMA 20', color='orange')
    ax3.legend()
    st.pyplot(fig3)

    st.write("**D. Volatility (20-Day Rolling Std)** – Risk management: periods of high/low volatility are key for real-world decisions.")

    if 'Volatility_20' in sample.columns:
        fig4, ax4 = plt.subplots()
        ax4.plot(sample['Date'], sample['Volatility_20'], color='red')
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Volatility")
        st.pyplot(fig4)

    st.markdown("> **Interpretation:** These plots help investors and researchers assess asset behavior, volatility, and risk—crucial for portfolio construction and asset selection. (Objective 1,3)")

# -------- TAB 2: ML Model Results --------
with tab2:
    st.header(f"Model Benchmarking for {market}")
    st.info("Upload your model results CSV (Market,Model,Accuracy,Precision,Recall,F1) or view the sample table below.")
    uploaded = st.file_uploader("Upload results CSV", type=['csv'])
    if uploaded:
        res = pd.read_csv(uploaded)
        st.dataframe(res)
        st.caption("This table summarizes model benchmarking. Accuracy near 0.5 = random guessing; higher = better predictive skill.")
    else:
        st.warning("No results uploaded yet. Sample results shown below.")
        st.table(pd.DataFrame(
            {'Model': ['RandomForest', 'SVM', 'ANN'],
             'Accuracy': [.50, .54, .51],
             'Note': ['Sample', 'Sample', 'Sample']}))
        st.caption("Update with your actual model results after running your benchmarks.")

    st.write("""
- **Interpretation**: Benchmarking multiple models (RF, SVM, ANN) addresses the research gap: little prior work compares models for *investment decision-making* (Objective 2).
- **Model performance table supports cross-market analysis (Objective 3)**
""")

# -------- TAB 3: Interpretation --------
with tab3:
    st.header("Cross-Market and Cross-Model Insights")
    st.markdown("""
- **ML models generally fail to beat random guessing for daily up/down prediction, even with advanced features.**
- **Similar results across S&P 500, Nifty 50, FTSE 100, and Bovespa highlight that this is not just a U.S./Europe phenomenon.**
- **This supports the view that financial markets are efficient and difficult to predict at short-term horizons.**
- **Your analysis demonstrates the *real-world effect* (or limitation) of ML models in actual investment settings, not just in-sample theory.**
- **Further work could analyze longer-term prediction, alternative data, or real portfolio backtesting.**

*Map each insight here to your objectives in your mid review discussion.*
""")

# -------- TAB 4: About / Research Objectives --------
with tab4:
    st.header("About this Research & Dashboard")
    st.markdown("""
#### **Research Gaps**
1. Most academic ML-in-finance research focuses on accuracy, not practical decision impact.
2. Few studies compare multiple ML models (RF, SVM, ANN) for real investment strategies.
3. Prior work is heavily U.S./Europe-centric; emerging markets get little attention.

#### **Objectives**
1. **Assess real-world effect of ML models** on investment/portfolio decisions (asset selection, risk management).
2. **Benchmark and compare** RF, SVM, ANN for investment-related prediction.
3. **Explore ML in emerging markets** (Nifty 50, Bovespa, etc.) and contrast with developed ones.

#### **How this dashboard supports the objectives:**
- Shows how features and models work for practical investment tasks (not just theory).
- Enables direct model comparison and cross-market benchmarking.
- Provides honest reporting—essential for academic rigor and real-world applicability.

*Update this text as you finalize your report/presentation!*
""")
