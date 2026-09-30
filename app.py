"""Веб-приложение ВКР: прогнозирование свойств композиционных материалов.

Запуск: streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# ----------------------------- загрузка моделей -----------------------------
@st.cache_resource
def load_models():
    model_elasticity = joblib.load('models/model_elasticity.joblib')
    model_strength = joblib.load('models/model_strength.joblib')
    model_nn = keras.models.load_model('models/nn_model.keras')
    scaler_nn = joblib.load('models/scaler_nn.joblib')
    return model_elasticity, model_strength, model_nn, scaler_nn


model_elasticity, model_strength, model_nn, scaler_nn = load_models()

# порядок колонок строго как при обучении
FEATURES_ML = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
               'модуль упругости, ГПа', 'Количество отвердителя, м.%',
               'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
               'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
               'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']
FEATURES_NN = [c for c in FEATURES_ML if c != 'Соотношение матрица-наполнитель']

# ----------------------------- интерфейс -----------------------------
st.title('Прогнозирование свойств композиционных материалов')
st.write('Введите свойства компонентов композита — приложение спрогнозирует '
         'конечные свойства материала и рекомендует соотношение '
         'матрица-наполнитель.')

with st.sidebar:
    st.header('Параметры композита')
    ratio = st.number_input('Соотношение матрица-наполнитель', 0.5, 10.0, 2.0, 0.01)
    density = st.number_input('Плотность, кг/м3', 1500.0, 2200.0, 2000.0, 1.0)
    elasticity_comp = st.number_input('Модуль упругости, ГПа', 100.0, 1000.0, 740.0, 0.1)
    hardener = st.number_input('Количество отвердителя, м.%', 20.0, 150.0, 100.0, 0.1)
    epoxy = st.number_input('Содержание эпоксидных групп, %', 15.0, 35.0, 22.0, 0.1)
    flash_temp = st.number_input('Температура вспышки, °C', 90.0, 420.0, 285.0, 0.1)
    surface_density = st.number_input('Поверхностная плотность, г/м2', 50.0, 1500.0, 210.0, 1.0)
    resin = st.number_input('Потребление смолы, г/м2', 50.0, 800.0, 220.0, 1.0)
    angle = st.selectbox('Угол нашивки, град', [0, 90])
    step = st.number_input('Шаг нашивки', 1.0, 20.0, 5.0, 0.5)
    patch_density = st.number_input('Плотность нашивки', 20.0, 120.0, 55.0, 1.0)

input_data = {
    'Соотношение матрица-наполнитель': ratio,
    'Плотность, кг/м3': density,
    'модуль упругости, ГПа': elasticity_comp,
    'Количество отвердителя, м.%': hardener,
    'Содержание эпоксидных групп,%_2': epoxy,
    'Температура вспышки, С_2': flash_temp,
    'Поверхностная плотность, г/м2': surface_density,
    'Потребление смолы, г/м2': resin,
    'Угол нашивки, град': 1 if angle == 90 else 0,  # бинарное кодирование
    'Шаг нашивки': step,
    'Плотность нашивки': patch_density,
}

if st.button('Рассчитать прогноз'):
    # --- прогноз свойств (случайный лес) ---
    X_ml = pd.DataFrame([input_data])[FEATURES_ML]
    pred_elasticity = model_elasticity.predict(X_ml)[0]
    pred_strength = model_strength.predict(X_ml)[0]

    # --- рекомендация соотношения (нейронная сеть) ---
    X_nn = pd.DataFrame([input_data])[FEATURES_NN]
    X_nn_scaled = scaler_nn.transform(X_nn)
    pred_ratio = model_nn.predict(X_nn_scaled, verbose=0)[0][0]

    st.subheader('Результаты прогноза')
    col1, col2, col3 = st.columns(3)
    col1.metric('Модуль упругости при растяжении', f'{pred_elasticity:.1f} ГПа')
    col2.metric('Прочность при растяжении', f'{pred_strength:.0f} МПа')
    col3.metric('Рекомендуемое соотношение матрица-наполнитель', f'{pred_ratio:.2f}')

    st.caption('Модели: случайный лес (свойства), нейронная сеть MLP (соотношение). '
               'Средняя ошибка моделей на тестовой выборке: около 2,4 ГПа, '
               '397 МПа и 0,75 единицы соотношения соответственно.')
