
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
st.title("ML in Finance – Multi‑Market Dashboard")

markets = {
    'S&P 500':'data/sp500_full_features.csv',
    'Nifty 50':'data/nifty50_full_features.csv',
    'FTSE 100':'data/ftse100_full_features.csv',
    'Bovespa':'data/bovespa_full_features.csv'
}

# Sidebar
st.sidebar.header("Controls")
market = st.sidebar.selectbox("Market", list(markets.keys()))
df = pd.read_csv(markets[market])

ticker = st.sidebar.selectbox("Ticker", sorted(df['Ticker'].unique()))
sample = df[df['Ticker']==ticker]

tab1, tab2, tab3 = st.tabs(["Data Visualisation","ML Results","Interpretation"])

with tab1:
    st.subheader("Closing Price")
    fig, ax = plt.subplots()
    ax.plot(sample['Date'], sample['Close'])
    ax.set_xlabel("Date"); ax.set_ylabel("Close")
    st.pyplot(fig)
    st.caption("Line chart shows historical closing price trend, useful for spotting overall direction and big moves.")

    st.subheader("Return Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(sample['Return'].dropna(), bins=50, color='steelblue')
    ax2.set_xlabel("Return"); ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    st.caption("Histogram depicts daily return distribution; heavy tails indicate higher probability of extreme moves.")

with tab2:
    st.info("Upload a CSV of your model results (Market,Model,Accuracy,Precision,Recall,F1) to populate the table.")
    uploaded = st.file_uploader("Upload results CSV", type=['csv'])
    if uploaded:
        res = pd.read_csv(uploaded)
        st.dataframe(res)
    else:
        st.warning("No results uploaded yet. Placeholder below.")
        st.table(pd.DataFrame(
            {'Model':['RandomForest','SVM','ANN'],'Accuracy':[.50,.54,.51],
             'Note':['Sample','Sample','Sample']}))

with tab3:
    st.write("""
**Key Findings**

* ML models struggle to outperform random guessing for daily direction prediction.
* Results are consistent across developed and emerging markets, supporting market‑efficiency view.
* Adding technical features and comparing RF, SVM, ANN shows limited incremental benefit.
""")
