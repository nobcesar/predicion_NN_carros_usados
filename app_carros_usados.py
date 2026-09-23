# app.py — Predicción de Precio Carros Usados UK
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ── Cargar modelo ─────────────────────────────────────────────────────────
modelo, min_max_scaler, variables = pickle.load(open('modeloNN-carros.pkl', 'rb'))

# ── Título ────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Predicción Carros Usados', page_icon='🚗', layout='centered')
st.title('🚗 Predicción de Precio — Carros Usados UK')
st.markdown('Ingresa las características del carro para estimar su precio en libras esterlinas (£)')

# ── Inputs ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    year       = st.slider('Año', min_value=1998, max_value=2020, value=2017, step=1)
    mileage    = st.number_input('Kilometraje (millas)', min_value=1, max_value=214000, value=20000, step=1000)
    tax        = st.number_input('Impuesto anual (£)', min_value=0, max_value=580, value=145, step=5)
    mpg        = st.number_input('Consumo (mpg)', min_value=5.0, max_value=200.0, value=50.0, step=0.5)
    engineSize = st.number_input('Tamaño motor (L)', min_value=0.6, max_value=6.6, value=2.0, step=0.1)

with col2:
    brand        = st.selectbox('Marca', ['Audi', 'BMW', 'Hyundai'])
    transmission = st.selectbox('Transmisión', ['Automatic', 'Manual', 'Semi-Auto'])
    fuelType     = st.selectbox('Combustible', ['Diesel', 'Hybrid', 'Petrol'])
    model_car    = st.selectbox('Modelo', sorted([
        '1 Series','2 Series','3 Series','4 Series','5 Series','6 Series','7 Series','8 Series',
        'A1','A2','A3','A4','A5','A6','A7','A8',
        'M2','M3','M4','M5','M6',
        'Q2','Q3','Q5','Q7','Q8',
        'R8','RS3','RS4','RS5','RS6','RS7',
        'S3','S4','S5','S8','SQ5','SQ7','TT',
        'X1','X2','X3','X4','X5','X6','X7',
        'Z3','Z4','i3','i8',
        'Santa Fe','Accent','Amica','Getz',
        'I10','I20','I30','I40','I800',
        'IX20','IX35','Ioniq','Kona',
        'Terracan','Tucson','Veloster'
    ]))

# ── Predicción ────────────────────────────────────────────────────────────
if st.button('💰 Predecir Precio'):

    # 1. Construir DataFrame
    data = pd.DataFrame([[year, mileage, tax, mpg, engineSize, brand, transmission, fuelType, model_car]],
                        columns=['year','mileage','tax','mpg','engineSize','brand','transmission','fuelType','model'])

    data_preparada = data.copy()

    # 2. Dummies (drop_first=False en despliegue)
    data_preparada = pd.get_dummies(
        data_preparada,
        columns=['brand','transmission','fuelType','model'],
        drop_first=False,
        dtype=int
    )

    # 3. Alinear columnas con las del entrenamiento
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    # 4. Normalizar numéricas (solo transform, NO fit)
    predictoras_numericas = ['year','mileage','tax','mpg','engineSize']
    data_preparada[predictoras_numericas] = min_max_scaler.transform(
        data_preparada[predictoras_numericas]
    )

    # 5. Predicción
    prediccion = modelo.predict(data_preparada)

    # 6. Resultado
    st.success(f'💷 Precio estimado: £{prediccion[0]:,.2f}')
    st.info(f'Equivalente aproximado en COP: ${prediccion[0] * 5200:,.0f}')
    st.warning('⚠️ El modelo tiene un error aproximado del 13.9% (MAPE)')
    st.dataframe(data)