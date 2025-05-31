import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Generar dataset determinístico
def generar_datos():
    X = []
    y = []
    for i in range(1, 101):
        for j in range(1, 101):
            X.append([i, j])
            y.append([i + j, i - j, i * j, i / (j + 1e-7)])
    return np.array(X), np.array(y)

X, y = generar_datos()

# Normalización por separado para cada salida
scaler_x = MinMaxScaler()
scalers_y = [MinMaxScaler() for _ in range(4)]

X = scaler_x.fit_transform(X)

y_normalizado = np.zeros_like(y)
for i in range(4):
    y_normalizado[:, i] = scalers_y[i].fit_transform(y[:, i].reshape(-1, 1)).ravel()

# Entrenamiento (70% entrenamiento, 30% validacion)
X_train, X_test, y_train, y_test = train_test_split(X, y_normalizado, test_size=0.3, random_state=42)

# Definir el modelo
modelo = Sequential([
    Dense(64, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='linear')
])

# ompilar y entrenar
modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
modelo.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluación del modelo
y_pred_norm = modelo.predict(X_test)
y_pred = np.zeros_like(y_pred_norm)
y_test_orig = np.zeros_like(y_pred_norm)

for i in range(4):
    y_pred[:, i] = scalers_y[i].inverse_transform(y_pred_norm[:, i].reshape(-1, 1)).ravel()
    y_test_orig[:, i] = scalers_y[i].inverse_transform(y_test[:, i].reshape(-1, 1)).ravel()

mae_global = mean_absolute_error(y_test_orig, y_pred)
print(f"Error promedio absoluto global (MAE): {mae_global:.4f}")

# Guardar resultados en CSV
X_test_orig = scaler_x.inverse_transform(X_test)

df_resultados = pd.DataFrame({
    'x1': X_test_orig[:, 0],
    'x2': X_test_orig[:, 1],

    'Suma_real': y_test_orig[:, 0],
    'Suma_predicha': y_pred[:, 0],
    'Error_suma': np.abs(y_test_orig[:, 0] - y_pred[:, 0]),

    'Resta_real': y_test_orig[:, 1],
    'Resta_predicha': y_pred[:, 1],
    'Error_resta': np.abs(y_test_orig[:, 1] - y_pred[:, 1]),

    'Multiplicacion_real': y_test_orig[:, 2],
    'Multiplicacion_predicha': y_pred[:, 2],
    'Error_multiplicacion': np.abs(y_test_orig[:, 2] - y_pred[:, 2]),

    'Division_real': y_test_orig[:, 3],
    'Division_predicha': y_pred[:, 3],
    'Error_division': np.abs(y_test_orig[:, 3] - y_pred[:, 3])
})

df_resultados.to_csv("resultados_modelo.csv", index=False)
print("Resultados guardados en 'resultados_modelo.csv'")

# Función para predecir valores
def predecir(x1, x2):
    entrada = scaler_x.transform(np.array([[x1, x2]]))
    pred = modelo.predict(entrada)[0]
    pred_inv = [scalers_y[i].inverse_transform([[pred[i]]])[0][0] for i in range(4)]
    print(f"\nPredicción para entrada [{x1}, {x2}]:")
    print(f"{x1} + {x2} ≈ {pred_inv[0]:.2f}")
    print(f"{x1} - {x2} ≈ {pred_inv[1]:.2f}")
    print(f"{x1} * {x2} ≈ {pred_inv[2]:.2f}")
    print(f"{x1} / {x2} ≈ {pred_inv[3]:.2f}")

# Prueba
predecir(25, 5)

