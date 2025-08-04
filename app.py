from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Data Visualization (Objective 1,3)",
    "2. ML Model Results (Objective 2,3)",
    "3. Interpretation (Objectives 1-3)",
    "4. About & Research Objectives"
])

# ... Tab 1 (Data Visualization) goes here ...

# -------- TAB 2: ML Model Results (LIVE) --------
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
