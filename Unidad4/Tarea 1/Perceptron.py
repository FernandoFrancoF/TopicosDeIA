# ========================================
# ENTRENAR PERCEPTRÓN PARA OPERACIONES
# ========================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Semilla para resultados reproducibles
np.random.seed(1)
tf.random.set_seed(1)

# ============================
# 1. Crear dataset artificial
# ============================
X = []
y_suma = []
y_resta = []
y_mult = []
y_div = []

for i in range(1, 101):
    for j in range(1, 101):
        X.append([i, j])
        y_suma.append(i + j)
        y_resta.append(i - j)
        y_mult.append(i * j)
        y_div.append(i / j)

X = np.array(X)
y_suma = np.array(y_suma)
y_resta = np.array(y_resta)
y_mult = np.array(y_mult)
y_div = np.array(y_div)

# ============================
# 2. Guardar dataset en CSV
# ============================
df = pd.DataFrame(X, columns=["num1", "num2"])
df["suma"] = y_suma
df["resta"] = y_resta
df["multiplicacion"] = y_mult
df["division"] = y_div
df.to_csv("dataset_operaciones.csv", index=False)
print("✅ Dataset guardado como 'dataset_operaciones.csv'")

# ============================
# 3. Normalización
# ============================
X_norm = X / 100.0
y_div_norm = y_div / 100.0  # Solo la división se normaliza

# ============================
# 4. Función para crear modelo
# ============================
def crear_modelo():
    model = Sequential([
        Dense(10, input_dim=2, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# ============================
# 5. Entrenamiento de modelos
# ============================
modelo_suma = crear_modelo()
modelo_suma.fit(X_norm, y_suma, epochs=100, verbose=0)

modelo_resta = crear_modelo()
modelo_resta.fit(X_norm, y_resta, epochs=100, verbose=0)

modelo_mult = crear_modelo()
modelo_mult.fit(X_norm, y_mult, epochs=100, verbose=0)

modelo_div = crear_modelo()
modelo_div.fit(X_norm, y_div_norm, epochs=100, verbose=0)

print("✅ Modelos entrenados correctamente.")

# ============================
# 6. Prueba con un ejemplo
# ============================
entrada = np.array([[25, 5]]) / 100.0

print("\n📊 Prueba con entrada: [25, 5]")
print("Suma:            ", modelo_suma.predict(entrada)[0][0])
print("Resta:           ", modelo_resta.predict(entrada)[0][0])
print("Multiplicación:  ", modelo_mult.predict(entrada)[0][0])
print("División:        ", modelo_div.predict(entrada)[0][0] * 100)  # desnormalizamos
