import streamlit as st 
import pandas as pd 
import joblib 
import numpy as np 

st.set_page_config(page_title="Guardian AI: Fraud Detector", page_icon="🛡️")

st.title("Guardian AI: Detección de Fraude")
st.markdown("Introduce los datos de la transacción para evaluar el riesgo.")

# Sidebar con info del proyecto
st.sidebar.header("Sistema de Control")
st.sidebar.info("Este modelo utiliza Random Forest entrenado con datos balanceados (SMOTE).")

# Formulario de entrada
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Monto de la Transacción ($)", min_value=0.0)
        distancia = st.number_input("Distancia al Comercio (km)", min_value=0.0)
    with col2:
        hora = st.slider("Hora del día", 0, 23, 12)
        edad = st.number_input("Edad del Cliente", 18, 100, 30)
    
    dia_semana = st.selectbox("Día de la semana", [0,1,2,3,4,5,6], 
                              format_func=lambda x: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][x])

    submit = st.form_submit_button("Evaluar Riesgo")

if submit:
    # Carga del modelo (debes haberlo guardado antes en el notebook)
    try:
        model = joblib.load('models/fraud_model.pkl')
        input_data = np.array([[amt, distancia, hora, dia_semana, edad]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"ALERTA: Probabilidad de FRAUDE detectada: {probability:.2%}")
        else:
            st.success(f"Transacción Legítima. Probabilidad de fraude: {probability:.2%}")
            
    except FileNotFoundError:
        st.warning(" Modelo no encontrado. Por favor, entrena y guarda el modelo en el notebook primero.")