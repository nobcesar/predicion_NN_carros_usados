# predicion_NN_carros_usados# 🚗 Predicción de Precio — Carros Usados UK

Aplicación web para estimar el precio de venta de carros usados en el mercado del Reino Unido, desarrollada como proyecto final del curso de Minería de Datos — Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana.

---

## 📌 Descripción

A partir de características técnicas y comerciales de un vehículo, el modelo predice su precio de venta en libras esterlinas (£). Se entrenaron y compararon cinco modelos de regresión bajo la metodología CRISP-DM, seleccionando la Red Neuronal Multicapa (MLP) como modelo final por obtener el menor error de predicción.

---

## 📊 Dataset

- **Fuente:** Carros usados Reino Unido
- **Registros:** 26,306
- **Variables:** año, kilometraje, impuesto, consumo (mpg), tamaño del motor, marca, transmisión, tipo de combustible y modelo
- **Variable objetivo:** precio (£)

---

## 🤖 Modelos evaluados

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Red Neuronal (MLP) | 2,733 | 4,038 | 13.9% ✅ |
| Random Forest | 3,438 | 4,924 | 17.3% |
| KNN | 3,871 | 5,531 | 19.1% |
| Árbol de Decisión | 4,076 | 5,735 | 20.4% |
| SVR | 6,790 | 8,942 | 37.7% |

Evaluación realizada con validación cruzada de 10 folds.

**Modelo seleccionado:** Red Neuronal MLP con `hidden_layer_sizes=(200, 100)`, `learning_rate_init=0.01`, `random_state=42`.

---

## 🗂️ Estructura del repositorio

📁 repo/
├── app.py # Aplicación Streamlit
├── modeloNN-carros.pkl # Modelo entrenado + scaler + variables
├── requirements.txt # Dependencias
└── README.md


---

## 🚀 Despliegue local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

---

## 🌐 Aplicación en línea

👉 [Ver aplicación en Streamlit](https://tu-app.streamlit.app)

---

## 🛠️ Tecnologías

- Python 3.10
- Scikit-learn
- Streamlit
- Pandas / NumPy

---

## 📋 Metodología

El proyecto sigue las fases de **CRISP-DM**:

1. **Entendimiento del negocio** — Estimar precios de venta en el mercado de segunda mano
2. **Entendimiento de los datos** — EDA con ProfileReport, correlaciones y distribuciones
3. **Preparación de datos** — Variables dummies, normalización MinMaxScaler
4. **Modelamiento** — 5 modelos comparados con validación cruzada (10 folds)
5. **Evaluación** — Selección por MAE, RMSE y MAPE
6. **Despliegue** — Aplicación interactiva en Streamlit

---

## ✍️ Autor

Desarrollado por **CESAR AUGUSTO MALDONADO PARRA** — Maestría en Ciencia de Datos, UPB 2026