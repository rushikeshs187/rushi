import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
import warnings

warnings.filterwarnings("ignore")

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

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'])
sample = df[df['Ticker'] == ticker].copy()

# -- For rolling beta, let user select the index ticker if present
if market == "S&P 500" and 'SPY' in df['Ticker'].unique():
    index_ticker = 'SPY'
elif market == "Nifty 50" and 'NIFTYBEES.NS' in df['Ticker'].unique():
    index_ticker = 'NIFTYBEES.NS'
else:
    index_ticker = None

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

    # Interactive date range
    st.write("**B. Interactive Date Range Selection** – Focus on any period.")
    min_date, max_date = sample['Date'].min(), sample['Date'].max()
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    mask = (sample['Date'] >= date_range[0]) & (sample['Date'] <= date_range[1])
    selected = sample[mask].copy()

    # Daily return histogram
    st.write("**C. Daily Return Distribution** – Shows market risk/volatility.")
    fig2, ax2 = plt.subplots()
    ax2.hist(selected['Return'].dropna(), bins=50, color='teal')
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # 20-day SMA
    st.write("**D. 20-Day SMA vs Close** – Visualize trend-following behavior.")
    fig3, ax3 = plt.subplots()
    ax3.plot(selected['Date'], selected['Close'], label='Close', color='navy')
    if 'SMA_20' in selected.columns:
        ax3.plot(selected['Date'], selected['SMA_20'], label='SMA 20', color='orange')
    ax3.legend()
    st.pyplot(fig3)

    # Volatility
    st.write("**E. Volatility (20-Day Rolling Std)** – Visualize regime shifts.")
    if 'Volatility_20' in selected.columns:
        fig4, ax4 = plt.subplots()
        ax4.plot(selected['Date'], selected['Volatility_20'], color='red')
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Volatility")
        st.pyplot(fig4)

    # Rolling mean/vol window
    st.write("**F. Rolling Mean & Rolling Volatility** – Adjust window for smoothing/risk.")
    window = st.slider("Rolling window (days)", min_value=5, max_value=100, value=20, step=5)
    selected['Rolling_Mean'] = selected['Close'].rolling(window).mean()
    selected['Rolling_Vol'] = selected['Return'].rolling(window).std()
    fig5, ax5 = plt.subplots()
    ax5.plot(selected['Date'], selected['Close'], label='Close', color='navy')
    ax5.plot(selected['Date'], selected['Rolling_Mean'], label=f'Rolling Mean ({window}d)', color='green')
    ax5.legend()
    st.pyplot(fig5)
    fig6, ax6 = plt.subplots()
    ax6.plot(selected['Date'], selected['Rolling_Vol'], color='crimson')
    ax6.set_xlabel("Date")
    ax6.set_ylabel(f"Volatility ({window}d std of returns)")
    st.pyplot(fig6)

    # Cumulative returns
    st.write("**G. Cumulative Returns** – See growth of $1 invested.")
    selected['Cumulative'] = (1 + selected['Return'].fillna(0)).cumprod()
    fig7, ax7 = plt.subplots()
    ax7.plot(selected['Date'], selected['Cumulative'], color='purple')
    ax7.set_xlabel("Date")
    ax7.set_ylabel("Cumulative Growth ($1 baseline)")
    st.pyplot(fig7)

    # Correlation heatmap
    st.write("**H. Correlation Heatmap** – See which features move together.")
    heatmap_features = [c for c in ['Close', 'Return', 'SMA_20', 'Volatility_20'] if c in selected.columns]
    corr_data = selected[heatmap_features].dropna().corr()
    fig8, ax8 = plt.subplots()
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap='coolwarm', ax=ax8)
    st.pyplot(fig8)
    st.caption("Correlation matrix shows linear dependence between features; high correlation suggests redundant features.")

    # Outlier detection
    st.write("**I. Outlier Detection** – Top 5 largest up/down days.")
    ups = selected.nlargest(5, 'Return')[['Date', 'Return']]
    downs = selected.nsmallest(5, 'Return')[['Date', 'Return']]
    st.write("Top 5 Positive Return Days:")
    st.table(ups)
    st.write("Top 5 Negative Return Days:")
    st.table(downs)
    st.caption("Outliers often correspond to major news or events.")

    # Drawdown plot
    st.write("**J. Drawdown Plot** – How much an investor could lose from peak.")
    running_max = selected['Close'].cummax()
    drawdown = (selected['Close'] - running_max) / running_max
    fig9, ax9 = plt.subplots()
    ax9.plot(selected['Date'], drawdown, color='brown')
    ax9.set_xlabel("Date")
    ax9.set_ylabel("Drawdown (relative)")
    st.pyplot(fig9)
    st.caption("Max drawdown is the worst peak-to-trough drop. Key risk metric for portfolio managers.")

    # Risk-return scatter
    st.write("**K. Risk-Return Scatterplot** – All tickers in this market.")
    means = df.groupby('Ticker')['Return'].mean()
    stds = df.groupby('Ticker')['Return'].std()
    fig10, ax10 = plt.subplots()
    ax10.scatter(stds, means, alpha=0.7)
    ax10.set_xlabel("Volatility (std)")
    ax10.set_ylabel("Mean Return")
    ax10.set_title("Risk-Return by Ticker")
    for tick in means.index:
        ax10.annotate(tick, (stds[tick], means[tick]), fontsize=7, alpha=0.7)
    st.pyplot(fig10)

    # Rolling beta to index (if possible)
    if index_ticker and ticker != index_ticker:
        st.write(f"**L. Rolling Beta to Index ({index_ticker})** – Time-varying correlation to market.")
        ticker_returns = selected.set_index('Date')['Return']
        index_returns = df[df['Ticker'] == index_ticker].set_index('Date')['Return'].reindex(selected['Date'])
        beta_win = st.slider("Beta window (days)", min_value=20, max_value=120, value=60, step=10)
        betas = []
        for i in range(len(ticker_returns)):
            if i < beta_win: betas.append(np.nan); continue
            y = ticker_returns.iloc[i-beta_win:i].values
            x = index_returns.iloc[i-beta_win:i].values
            if np.isnan(y).any() or np.isnan(x).any(): betas.append(np.nan); continue
            beta = np.polyfit(x, y, 1)[0]
            betas.append(beta)
        fig11, ax11 = plt.subplots()
        ax11.plot(selected['Date'], betas, color='darkorange')
        ax11.set_xlabel("Date")
        ax11.set_ylabel(f"Beta (window={beta_win})")
        st.pyplot(fig11)
        st.caption(f"Rolling beta measures how much {ticker} moves relative to {index_ticker}. Beta>1: more volatile than market.")

    # Autocorrelation plot
    st.write("**M. Autocorrelation of Returns** – Predictability check.")
    acf_lags = st.slider("Autocorrelation lags", min_value=5, max_value=60, value=20, step=5)
    acf_vals = acf(selected['Return'].dropna(), nlags=acf_lags)
    fig12, ax12 = plt.subplots()
    ax12.bar(range(acf_lags+1), acf_vals, color='dodgerblue')
    ax12.set_xlabel("Lag")
    ax12.set_ylabel("Autocorrelation")
    ax12.set_title("Autocorrelation of Returns")
    st.pyplot(fig12)
    st.caption("If autocorrelation is close to zero for all lags, returns are not predictable (efficient market hypothesis).")

    # Missing data
    st.write("**N. Missing Data Check** – Columns with most missing values.")
    missing = selected.isnull().astype(int)
    st.dataframe(missing.sum().reset_index().rename(columns={0:'MissingCount'}))

    st.markdown("> **Interpretation:** This comprehensive EDA suite supports all research objectives, giving you the data depth and analytical tools needed for top-tier finance research.")

# (Leave tabs 2, 3, 4 unchanged from earlier – let me know if you want the full file with those included!)
