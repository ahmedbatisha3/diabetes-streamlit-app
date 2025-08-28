import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Diabetes Prediction App")

uploaded_file = st.file_uploader("Upload your Diabetes CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Sample")
    st.write(df.head())

    st.subheader("Data Visualization")
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    sns.histplot(df['AGE'], kde=True, bins=30, color="skyblue", ax=axes[0])
    axes[0].set_title("Distribution of Age")
    sns.histplot(df['BMI'], bins=30, kde=True, color="lightgreen", ax=axes[1])
    axes[1].set_title("BMI Distribution")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include=["int64","float64"]).corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    df = df.drop(columns=['ID', 'No_Pation'], errors='ignore')
    df = df.fillna(df.mean(numeric_only=True))
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['CLASS'] = df['CLASS'].fillna(df['CLASS'].mode()[0])

    df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
    df['CLASS'] = df['CLASS'].astype(str).str.strip()
    df = df.drop_duplicates()

    X = df.drop(columns=['CLASS'])
    y = df['CLASS']

    le_gender = LabelEncoder()
    le_target = LabelEncoder()
    if 'Gender' in X.columns:
        X.loc[:, 'Gender'] = le_gender.fit_transform(X['Gender'])
    y = le_target.fit_transform(y)

    sss = ['AGE','Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI']
    sss = [c for c in sss if c in X.columns]  # ignore missing cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train[sss])
    X_train.loc[:, sss] = scaler.transform(X_train[sss])
    X_test.loc[:, sss] = scaler.transform(X_test[sss])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Evaluation")
    acc = accuracy_score(y_test, y_pred)
    st.write(f" Accuracy: **{acc:.2f}**")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                ax=ax)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le_target.classes_))

    st.subheader("🔮 Predict Patient Status (Manual Input)")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        urea = st.number_input("Urea", value=25.0)
        cr = st.number_input("Cr", value=1.0)
        hba1c = st.number_input("HbA1c", value=5.5)
        chol = st.number_input("Cholesterol", value=180.0)
        tg = st.number_input("Triglycerides (TG)", value=150.0)
        hdl = st.number_input("HDL", value=45.0)
        ldl = st.number_input("LDL", value=100.0)
        vldl = st.number_input("VLDL", value=30.0)
        bmi = st.number_input("BMI", value=25.0)
        gender = st.selectbox("Gender", ["M", "F"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            "AGE": [age],
            "Urea": [urea],
            "Cr": [cr],
            "HbA1c": [hba1c],
            "Chol": [chol],
            "TG": [tg],
            "HDL": [hdl],
            "LDL": [ldl],
            "VLDL": [vldl],
            "BMI": [bmi],
            "Gender": [le_gender.transform([gender])[0]]
        })

        input_data = input_data[X_train.columns]

        input_data.loc[:, sss] = scaler.transform(input_data[sss])

        prediction = model.predict(input_data)[0]
        predicted_label = le_target.inverse_transform([prediction])[0]

        if predicted_label.lower().startswith("n"):
            st.success(f"✅ The model predicts: {predicted_label}")
        else:
            st.error(f"⚠️ The model predicts: {predicted_label}")

else:
    st.info("Please upload your dataset CSV file to start analysis")
