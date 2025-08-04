import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("ML in Finance – Multi-Market Dashboard")

# --------- Update: files in root, no data/ prefix ----------
markets = {
    'S&P 500':'sp500_full_features.csv',
    'Nifty 50':'nifty50_full_features.csv',
    'FTSE 100':'ftse100_full_features.csv',
    'Bovespa':'bovespa_full_features.csv'
}
# ------------------------------------------------------------

# Sidebar controls
st.sidebar.header("Controls")
market = st.sidebar.selectbox("Market", list(markets.keys()))
df = pd.read_csv(markets[market])

ticker = st.sidebar.selectbox("Ticker", sorted(df['Ticker'].unique()))
sample = df[df['Ticker'] == ticker]

tab1, tab2, tab3 = st.tabs(["Data Visualisation", "ML Results", "Interpretation"])

with tab1:
    st.subheader("Closing Price")
    fig, ax = plt.subplots()
    ax.plot(sample['Date'], sample['Close'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    st.pyplot(fig)
    st.caption("Line chart shows historical closing price trend, useful for spotting overall direction and big moves.")

    st.subheader("Return Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(sample['Return'].dropna(), bins=50, color='steelblue')
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    st.caption("Histogram depicts daily return distribution; heavy tails indicate higher probability of extreme moves.")

    st.subheader("20-Day SMA vs Close")
    fig3, ax3 = plt.subplots()
    ax3.plot(sample['Date'], sample['Close'], label='Close')
    if 'SMA_20' in sample.columns:
        ax3.plot(sample['Date'], sample['SMA_20'], label='SMA 20')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend()
    st.pyplot(fig3)
    st.caption("Shows how the 20-day moving average smooths price swings and helps visualize trends.")

    st.subheader("Volatility (20-Day Rolling Std)")
    if 'Volatility_20' in sample.columns:
        fig4, ax4 = plt.subplots()
        ax4.plot(sample['Date'], sample['Volatility_20'])
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Volatility")
        st.pyplot(fig4)
        st.caption("Rolling volatility shows periods of market calm and turbulence. Useful for risk analysis.")

with tab2:
    st.info("Upload a CSV of your model results (Market,Model,Accuracy,Precision,Recall,F1) to populate the table below.")
    uploaded = st.file_uploader("Upload results CSV", type=['csv'])
    if uploaded:
        res = pd.read_csv(uploaded)
        st.dataframe(res)
        st.caption("This table summarizes your model benchmarking across markets/models. Higher accuracy means better next-day prediction, but note: 50% accuracy is random guessing.")
    else:
        st.warning("No results uploaded yet. Sample table below.")
        st.table(pd.DataFrame(
            {'Model': ['RandomForest', 'SVM', 'ANN'], 'Accuracy': [.50, .54, .51],
             'Note': ['Sample', 'Sample', 'Sample']}))
        st.caption("Update with your actual results after running your models.")

with tab3:
    st.markdown("""
    ### Key Findings & Interpretation

    - **ML models (Random Forest, SVM, ANN) generally struggle to beat random guessing for daily up/down prediction across all markets.**
    - **Adding features like lagged returns, SMA, volatility, or fundamentals gives only modest improvements.**
    - **Results are similar across S&P 500, Nifty 50, FTSE 100, Bovespa, supporting the view that liquid markets are highly efficient at this time frame.**
    - **Interpreting the results: Honest reporting of model accuracy, limitations, and cross-market comparison gives real-world insight, not just academic theory.**

    ---
    Edit or expand this text to match your own findings and insights!
    """)
