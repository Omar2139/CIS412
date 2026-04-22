import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="CIS412 Customer Response Predictor", layout="wide")

st.title("📊 CIS412 — Customer Marketing Response Predictor")
st.markdown("Upload your `superstore_data.csv` to train and compare classification models.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload superstore_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.expander("🔍 Raw Data Preview"):
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
        st.write(df.describe())

    # --- Preprocessing ---
    st.subheader("🛠️ Data Preprocessing")

    df['Income'] = df['Income'].fillna(df['Income'].median())

    current_year = datetime.datetime.now().year
    df['Age'] = current_year - df['Year_Birth']
    df = df.drop('Year_Birth', axis=1)
    df = df[(df['Age'] > 18) & (df['Age'] < 90)]

    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Customer_Tenure'] = (pd.Timestamp.now() - df['Dt_Customer']).dt.days
    df = df.drop('Dt_Customer', axis=1)

    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spending'] = df[spending_cols].sum(axis=1)

    df['Total_Purchases'] = (
        df['NumWebPurchases'] +
        df['NumCatalogPurchases'] +
        df['NumStorePurchases']
    )

    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

    st.success("✅ Preprocessing complete.")
    st.write(f"Final dataset shape: `{df.shape}`")

    # --- Class Imbalance ---
    st.subheader("⚖️ Class Imbalance in Response")
    counts = df['Response'].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    counts.plot(kind='bar', ax=ax1, color=['#4C72B0', '#DD8452'])
    ax1.set_title('Class Imbalance in Response')
    ax1.set_xlabel('Response Class')
    ax1.set_ylabel('Number of Customers')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['0 = No', '1 = Yes'], rotation=0)
    st.pyplot(fig1)

    # --- Train/Test Split ---
    X = df.drop('Response', axis=1)
    y = df['Response']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model Selection ---
    st.subheader("🤖 Train Models")
    model_choice = st.multiselect(
        "Select models to train:",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

    if st.button("🚀 Train Selected Models"):
        results = {}

        if "Logistic Regression" in model_choice:
            with st.spinner("Training Logistic Regression..."):
                lr = LogisticRegression(max_iter=1000, class_weight='balanced')
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                results["Logistic Regression"] = (lr, y_pred_lr)

        if "Random Forest" in model_choice:
            with st.spinner("Training Random Forest..."):
                rf = RandomForestClassifier(class_weight='balanced', random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                results["Random Forest"] = (rf, y_pred_rf)

        if "Gradient Boosting" in model_choice:
            with st.spinner("Training Gradient Boosting..."):
                gb = GradientBoostingClassifier(random_state=42)
                gb.fit(X_train, y_train)
                y_pred_gb = gb.predict(X_test)
                results["Gradient Boosting"] = (gb, y_pred_gb)

        # --- Results ---
        st.subheader("📈 Model Results")

        for model_name, (model, y_pred) in results.items():
            with st.expander(f"📋 {model_name} Results", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Classification Report**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    st.dataframe(report_df)

                with col2:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax_cm, colorbar=False)
                    ax_cm.set_title(model_name)
                    st.pyplot(fig_cm)

        # --- Feature Importance (Gradient Boosting) ---
        if "Gradient Boosting" in results:
            st.subheader("🔑 Top Features — Gradient Boosting")
            gb_model = results["Gradient Boosting"][0]
            importance_gb = pd.Series(gb_model.feature_importances_, index=X.columns)
            top10 = importance_gb.sort_values(ascending=False).head(10)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            top10.sort_values().plot(kind='barh', ax=ax2, color='steelblue')
            ax2.set_title("Top 10 Features Influencing Customer Response")
            st.pyplot(fig2)

else:
    st.info("👆 Please upload your `superstore_data.csv` file to get started.")
