import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML in Finance – Research Dashboard", layout="wide")

markets = {
    'S&P 500': 'sp500_full_features.csv',
    'Nifty 50': 'nifty50_full_features.csv',
    'FTSE 100': 'ftse100_full_features.csv',
    'Bovespa': 'bovespa_full_features.csv'
}

st.sidebar.header("Dashboard Controls")
market = st.sidebar.selectbox("Select Market", list(markets.keys()), help="Choose which market to analyze.")
df = pd.read_csv(markets[market])
ticker = st.sidebar.selectbox("Select Ticker", sorted(df['Ticker'].unique()), help="Choose company to visualize.")
sample = df[df['Ticker'] == ticker].copy()

st.title("Machine Learning in Finance: Multi-Market Research Dashboard")
st.markdown("""
**This dashboard is built to support research objectives:**
- *Assess real-world impact of ML models on investment decisions (Objective 1)*
- *Benchmark and compare Random Forest, SVM, ANN (Objective 2)*
- *Compare developed vs. emerging markets (Objective 3)*
- *Provide transparent, empirical findings—not just theoretical accuracy*
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Data Visualization (Objective 1,3)",
    "2. ML Model Results (Objective 2,3)",
    "3. Interpretation (Objectives 1-3)",
    "4. About & Research Objectives"
])

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

    # ----- ADVANCED EDA -----
    st.write("**E. Rolling Mean & Rolling Volatility** – Use sliders to adjust the window size and observe smoothing/risk.")
    window = st.slider("Select rolling window (days)", min_value=5, max_value=100, value=20, step=5)
    sample['Rolling_Mean'] = sample['Close'].rolling(window).mean()
    sample['Rolling_Vol'] = sample['Return'].rolling(window).std()

    fig5, ax5 = plt.subplots()
    ax5.plot(sample['Date'], sample['Close'], label='Close', color='navy')
    ax5.plot(sample['Date'], sample['Rolling_Mean'], label=f'Rolling Mean ({window}d)', color='green')
    ax5.legend()
    st.pyplot(fig5)
    st.caption(f"Rolling mean helps smooth noise; window size controls sensitivity. Larger = smoother.")

    fig6, ax6 = plt.subplots()
    ax6.plot(sample['Date'], sample['Rolling_Vol'], color='crimson')
    ax6.set_xlabel("Date")
    ax6.set_ylabel(f"Volatility ({window}d std of returns)")
    st.pyplot(fig6)
    st.caption(f"Rolling volatility (std of returns over {window} days): higher = more risk.")

    st.write("**F. Cumulative Returns** – See growth of $1 invested (ignoring dividends/fees).")
    sample['Cumulative'] = (1 + sample['Return'].fillna(0)).cumprod()
    fig7, ax7 = plt.subplots()
    ax7.plot(sample['Date'], sample['Cumulative'], color='purple')
    ax7.set_xlabel("Date")
    ax7.set_ylabel("Cumulative Growth ($1 baseline)")
    st.pyplot(fig7)
    st.caption("Shows how an initial $1 would have grown, making it easy to compare long-run performance.")

    st.write("**G. Risk-Return Scatterplot** – Visualize mean return vs. volatility for all tickers (market risk/return tradeoff).")
    means = df.groupby('Ticker')['Return'].mean()
    stds = df.groupby('Ticker')['Return'].std()
    fig8, ax8 = plt.subplots()
    ax8.scatter(stds, means, alpha=0.7)
    ax8.set_xlabel("Volatility (std)")
    ax8.set_ylabel("Mean Return")
    ax8.set_title("Risk-Return by Ticker")
    for tick in means.index:
        ax8.annotate(tick, (stds[tick], means[tick]), fontsize=7, alpha=0.7)
    st.pyplot(fig8)
    st.caption("Risk-Return chart helps spot tickers with high risk or high average return.")

    st.write("**H. Interactive Histogram** – Choose bins for return histogram.")
    bins = st.slider("Histogram bins", min_value=10, max_value=100, value=50, step=10)
    fig9, ax9 = plt.subplots()
    ax9.hist(sample['Return'].dropna(), bins=bins, color='skyblue')
    st.pyplot(fig9)
    st.caption("Adjust bin count to see finer or coarser return distribution.")

    st.write("**I. Missing Data Check** – See which columns have missing values.")
    missing = sample.isnull().astype(int)
    st.dataframe(missing.sum().reset_index().rename(columns={0:'MissingCount'}))
    st.caption("Columns with more missing values may require cleaning or careful analysis.")

    st.markdown("> **Interpretation:** These advanced plots help answer deeper research questions about risk, return, stability, and data quality—adding rigor and supporting robust portfolio analysis (Objective 1).")

with tab2:
    st.header(f"Model Benchmarking for All Markets")
    st.info("Click the button below to automatically run RandomForest, SVM, and ANN on each market. Results will appear in the table and can be downloaded for your report.")

    if st.button("Run All ML Benchmarks"):
        results = []
        progress = st.progress(0)
        for i, (mkt, file) in enumerate(markets.items()):
            dfm = pd.read_csv(file)
            dfm['Return_1d_ago'] = dfm.groupby('Ticker')['Return'].shift(1)
            dfm['Return_5d_ago'] = dfm.groupby('Ticker')['Return'].shift(5)
            dfm['SMA_50'] = dfm.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
            dfm['Target'] = (dfm.groupby('Ticker')['Return'].shift(-1) > 0).astype(int)
            dfm = dfm.dropna(subset=['Return', 'Return_1d_ago', 'Return_5d_ago', 'SMA_20', 'SMA_50', 'Volatility_20', 'Target'])
            features = ['Return', 'Return_1d_ago', 'Return_5d_ago', 'SMA_20', 'SMA_50', 'Volatility_20']
            X = dfm[features]
            y = dfm['Target']
            dfm['Date'] = pd.to_datetime(dfm['Date'])
            train = dfm[dfm['Date'] < '2023-01-01']
            test = dfm[dfm['Date'] >= '2023-01-01']
            X_train = train[features]; y_train = train['Target']
            X_test = test[features]; y_test = test['Target']

            # RandomForest
            model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            results.append({
                'Market': mkt, 'Model': 'RandomForest',
                'Accuracy': accuracy_score(y_test, y_pred_rf),
                'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
                'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
                'F1': f1_score(y_test, y_pred_rf, zero_division=0)})

            # SVM
            model_svm = SVC()
            model_svm.fit(X_train, y_train)
            y_pred_svm = model_svm.predict(X_test)
            results.append({
                'Market': mkt, 'Model': 'SVM',
                'Accuracy': accuracy_score(y_test, y_pred_svm),
                'Precision': precision_score(y_test, y_pred_svm, zero_division=0),
                'Recall': recall_score(y_test, y_pred_svm, zero_division=0),
                'F1': f1_score(y_test, y_pred_svm, zero_division=0)})

            # ANN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
            model_ann.fit(X_train_scaled, y_train)
            y_pred_ann = model_ann.predict(X_test_scaled)
            results.append({
                'Market': mkt, 'Model': 'ANN',
                'Accuracy': accuracy_score(y_test, y_pred_ann),
                'Precision': precision_score(y_test, y_pred_ann, zero_division=0),
                'Recall': recall_score(y_test, y_pred_ann, zero_division=0),
                'F1': f1_score(y_test, y_pred_ann, zero_division=0)})

            progress.progress((i+1)/len(markets))

        resdf = pd.DataFrame(results)
        st.dataframe(resdf)
        st.download_button("Download Results as CSV", resdf.to_csv(index=False), file_name="ml_model_results.csv")
        st.success("Benchmarks complete! Results table updated. These results are directly relevant to Objective 2 (model comparison) and Objective 3 (market benchmarking).")

    else:
        st.info("Click 'Run All ML Benchmarks' to benchmark RandomForest, SVM, ANN for all markets and display results here. You can also download the summary.")

    st.write("""
    - **Interpretation**: Live benchmarking addresses the research gap on comparative model performance (Objective 2) and enables easy cross-market analysis (Objective 3).
    - **No manual file uploads required – everything runs and appears instantly!**
    """)

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
