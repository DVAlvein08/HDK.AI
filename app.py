import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

@st.cache_data
def train_model():
    df = pd.read_csv("mo_hinh_AI.csv")
    df = df[df["tác nhân"] != "unspecified"]
    X = df.drop(columns=["tác nhân"])
    y = df["tác nhân"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model, X.columns.tolist(), label_encoder

@st.cache_data
def load_resistance_data():
    return pd.read_csv("mo_hinh_KSD.csv")

def suggest_antibiotics(pathogen, resistance_df):
    if pathogen == "RSV":
        return []
    elif pathogen == "M. pneumonia":
        return ["Macrolides", "Doxycycline", "Fluoroquinolones"]
    else:
        row = resistance_df[resistance_df["Tác nhân"] == pathogen]
        if row.empty:
            return []
        antibiotics = row.drop(columns=["Tác nhân"]).T
        sensitive = antibiotics[antibiotics[antibiotics.columns[0]] >= 0.5]
        return sensitive.index.tolist()

st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

model, feature_cols, label_encoder = train_model()
resistance_df = load_resistance_data()

st.header("📋 Nhập dữ liệu lâm sàng")
user_input = {}
for col in feature_cols:
    user_input[col] = st.number_input(col, value=0.0, step=1.0)

if st.button("🔍 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    predicted_pathogen = label_encoder.inverse_transform([prediction])[0]
    st.success(f"Tác nhân gây bệnh được dự đoán: **{predicted_pathogen}**")

    st.subheader("💊 Kháng sinh gợi ý:")
    antibiotics = suggest_antibiotics(predicted_pathogen, resistance_df)
    if antibiotics:
        for ab in antibiotics:
            st.markdown(f"- {ab}")
    else:
        st.info("Không có kháng sinh nào được gợi ý.")