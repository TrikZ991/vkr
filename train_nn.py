"""Обучение нейронной сети, рекомендующей соотношение матрица-наполнитель.

Скрипт повторяет раздел 2.4 ноутбука: загружает данные, обучает MLP
и сохраняет модель и скейлер в папку models/ для приложения app.py.

Запуск: python train_nn.py
"""
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers

RANDOM_STATE = 42
keras.utils.set_random_seed(RANDOM_STATE)

# ----------------------------- данные -----------------------------
df1 = pd.read_excel('data/X_bp.xlsx')
df2 = pd.read_excel('data/X_nup.xlsx')
df = df1.merge(df2, left_index=True, right_index=True,
               how='inner', suffixes=('', '_2'))
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
df['Угол нашивки, град'] = df['Угол нашивки, град'].replace(90, 1)
df = df[df['Плотность нашивки'] > 0].reset_index(drop=True)

# целевая — соотношение матрица-наполнитель;
# из признаков исключаем целевые переменные задачи регрессии
y = df['Соотношение матрица-наполнитель']
X = df.drop(columns=['Соотношение матрица-наполнитель',
                     'Модуль упругости при растяжении, ГПа',
                     'Прочность при растяжении, МПа'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------- модель -----------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mae')

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(X_train, y_train, epochs=300, batch_size=32,
          validation_split=0.2, callbacks=[early_stopping], verbose=1)

mae = np.mean(np.abs(model.predict(X_test).flatten() - y_test))
print(f'MAE на тестовой выборке: {mae:.3f}')

# ----------------------------- сохранение -----------------------------
model.save('models/nn_model.keras')
joblib.dump(scaler, 'models/scaler_nn.joblib')
print('Модель и скейлер сохранены в models/')
