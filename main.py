# =============================================================================
# Copyright (c) [2024] [Bastian Muñoz Ordenes]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================


import logging
import os
import pickle
import sys
from pathlib import Path
from statistics import LinearRegression

import pandas as pd
from joblib import load
from keras.src.regularizers import regularizers
from matplotlib import pyplot as plt
from pygments.lexers import go

from libs.benchmark import Benchmark
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

log = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@st.cache_data
def load_model():
    # Verifica si el archivo existe
    file_path = os.path.abspath('output/model.pkl')
    if not os.path.isfile(file_path):
        print(f"El archivo {file_path} no existe.")
    else:
        print(f"El archivo {file_path} existe.")

    model_path = Path("output/model.pkl")
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)

def train_model():
    # Cargar los datos
    df = pd.read_excel('../data/datos_finales.xlsx')

    # Convertir la columna 'Fecha' a características útiles
    df['FECHA (YYMMDD)'] = pd.to_datetime(df['FECHA (YYMMDD)'])
    df['Mes'] = df['FECHA (YYMMDD)'].dt.month
    df['Dia'] = df['FECHA (YYMMDD)'].dt.day
    df['Año'] = df['FECHA (YYMMDD)'].dt.year

    # Seleccionar las columnas que usarás como características y objetivo
    features = df[['Temperatura media', 'Mes', 'Dia', 'Año']]
    targets = df[['Registros validos mp10', 'Registros validos mp2_5', 'Registros validos so2']]

    # Lidiar con valores faltantes (si es necesario)
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    targets_imputed = imputer.fit_transform(targets)

    # Escalar las características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, targets_imputed, test_size=0.2,
                                                        random_state=42)

    # Crear un modelo de regresión lineal (puedes probar otros modelos)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(x_test)

    # Evaluar el modelo
    for i, target_name in enumerate(['mp10', 'mp2_5', 'so2']):
        print(f"\nEvaluación para {target_name}:")
        print(f"Error cuadrático medio: {mean_squared_error(y_test[:, i], y_pred[:, i])}")
        print(f"R2 score: {r2_score(y_test[:, i], y_pred[:, i])}")

    # Guardar el modelo en un archivo
    with open("output/model.pkl", "wb") as file:
        pickle.dump(model, file)
    log.debug("Model saved successfully.")

def the_main(sns=None):
    # Cargar los datos
    df = pd.read_excel('/data/datos_finales.xlsx')

    print(f"Using current path: {Path.cwd()} !!")
    # Load the pre-trained model
    model = load_model()
    # Set up the Streamlit app interface
    st.title("Prediccion por fecha")
    st.markdown("Utilizamos una regresion lineal para tratar de predecir mp10, mp2.5 y so2 respecto a una fecha y una temporada")
    st.header("Fecha y temperatura")
    col1,col2 = st.columns(2)
    # Input sliders for sepal characteristics
    with col1:
        st.text("Fecha")
        dia = st.slider("Dia", 1, 31, 1)
        mes = st.slider("Mes", 1, 12, 1)
        anio = st.number_input("Año:", min_value=2018, max_value=2030)

    with col2:
        st.text("Temperatura")
        temperatura = st.slider("Temperatura", -15.0, 30.0, 0.1)


    # Create a DataFrame with the input features
    input_features = pd.DataFrame(
        [[temperatura, mes, dia, anio]],
        columns=["temperatura", "mes", "dia", "anio"],
    )

    # Escalar las características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(input_features)

    # Predict the class of the iris flower based on the input features
    prediction = model.predict(features_scaled)
    pred = prediction[0]

    st.write(f"Prediccion de mp10: {pred[0]}")
    st.write(f"Prediccion de mp2.5: {pred[1]}")
    st.write(f"Prediccion de so2: {pred[2]}")

    st.write(input_features)

    fig = go.Figure()

    # Agregar líneas para cada variable
    fig.add_trace(
        go.Scatter(x=df['FECHA (YYMMDD)'], y=df['Registros validos mp10'], mode='lines', name='MP10'))
    fig.add_trace(
        go.Scatter(x=df['FECHA (YYMMDD)'], y=df['Registros validos mp2_5'], mode='lines', name='MP2.5'))
    fig.add_trace(
        go.Scatter(x=df['FECHA (YYMMDD)'], y=df['Registros validos so2'], mode='lines', name='SO2'))

    # Añadir detalles al gráfico
    fig.update_layout(
        title='Concentraciones de Contaminantes a lo Largo del Tiempo',
        xaxis_title='Fecha',
        yaxis_title='Concentración',
        legend_title='Contaminantes',
        template='plotly_dark'
    )

    # Mostrar el gráfico
    fig.show()


if __name__ == '__main__':
    the_main()
