import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("🔮 Application de Prédiction ML")
x = st.number_input("Entrez une valeur :", min_value=0.0, step=1.0)

if st.button("Prédire"):
    prediction = model.predict([[x]])
    st.success(f"Résultat : {prediction[0]:.2f}")
