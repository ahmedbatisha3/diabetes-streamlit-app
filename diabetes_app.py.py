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


    df = df.drop(columns=['ID', 'No_Pation'])
    df = df.fillna(df.mean(numeric_only=True))
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['CLASS'] = df['CLASS'].fillna(df['CLASS'].mode()[0])
    df['CLASS'] = df['CLASS'].str.strip()
    df['Gender'] = df['Gender'].replace('f', 'F')
    df = df.drop_duplicates()

    X = df.drop(columns=['CLASS'])
    y = df['CLASS']

    le = LabelEncoder()
    X.loc[:, 'Gender'] = le.fit_transform(X['Gender'])
    y = le.fit_transform(y)

    sss = ['AGE','Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI']
    scaler = StandardScaler()
    X.loc[:, sss] = scaler.fit_transform(X[sss])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    st.subheader("Model Evaluation")
    acc = accuracy_score(y_test, y_pred)
    st.write(f" Accuracy: **{acc:.2f}**")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Normal (0)", "Patient (1)"],
                yticklabels=["Normal (0)", "Patient (1)"],
                ax=ax)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload your dataset CSV file to start analysis")
