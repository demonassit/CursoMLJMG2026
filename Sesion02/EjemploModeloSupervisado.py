# Ejemplo de modelo supervisado: comparamos K vecinos (KNN) contra regresion lineal
# Idea: predecir que tanto avanza la diabetes de un paciente un año después,
# a partir de 10 medidas clinicas (edad, sexo, indice de masa corporal, presion, etc.)
# Flujo: cargar datos -> dividir 80/20 -> escalar -> entrenar -> predecir -> comparar

from sklearn.datasets import load_diabetes
# dataset de 442 pacientes con 10 caracteristicas, ya viene incluido en sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# KNN mide distancias, por eso necesitamos que todas las variables esten en la misma escala
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

# 1. Cargamos los datos
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target  # y = progresion de la enfermedad (valor entre 25 y 346)
print("Caracteristicas:", diabetes.feature_names)
print("Forma de X:", X.shape)

# 2. Dividimos 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Escalamos: el scaler aprende media y desviacion SOLO con los datos de entrenamiento
scaler = StandardScaler()
X_train_esc = scaler.fit_transform(X_train)
X_test_esc = scaler.transform(X_test)

# 4. Buscamos el mejor numero de vecinos k
# con k muy chico el modelo memoriza (sobreajuste), con k muy grande se vuelve demasiado simple
valores_k = range(1, 31)
r2_por_k = []
for k in valores_k:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_esc, y_train)
    r2_por_k.append(r2_score(y_test, knn.predict(X_test_esc)))

mejor_k = valores_k[int(np.argmax(r2_por_k))]
print(f"\nMejor k encontrado: {mejor_k}")
# nota: en un proyecto real k se elige con validacion cruzada, no con el conjunto de prueba

# 5. Entrenamos los dos modelos finales
modelo_knn = KNeighborsRegressor(n_neighbors=mejor_k)
modelo_knn.fit(X_train_esc, y_train)
y_pred_knn = modelo_knn.predict(X_test_esc)

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train_esc, y_train)
y_pred_lineal = modelo_lineal.predict(X_test_esc)

# 6. Evaluamos: MSE (error, mientras mas bajo mejor) y R2 (1.0 = prediccion perfecta)
resultados = {
    f"KNN (k={mejor_k})": y_pred_knn,
    "Regresion lineal": y_pred_lineal,
}
print("\nModelo                 MSE        R2")
for nombre, y_pred in resultados.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{nombre:<20} {mse:>9.2f}   {r2:>6.3f}")

# la regresion lineal nos dice cuanto pesa cada variable (coeficientes)
print("\nCoeficientes de la regresion lineal:")
for nombre, coef in zip(diabetes.feature_names, modelo_lineal.coef_):
    print(f"  {nombre:<5} {coef:>8.2f}")

# 7. Graficamos
fig, ejes = plt.subplots(1, 3, figsize=(16, 5))

# grafica 1: como cambia R2 segun k
ejes[0].plot(valores_k, r2_por_k, marker="o")
ejes[0].axvline(mejor_k, color="red", linestyle="--", label=f"mejor k = {mejor_k}")
ejes[0].set_title("KNN: R2 segun numero de vecinos")
ejes[0].set_xlabel("k")
ejes[0].set_ylabel("R2 en prueba")
ejes[0].legend()

# graficas 2 y 3: valor real vs predicho, si el modelo fuera perfecto todos los puntos caerian en la diagonal
for eje, (nombre, y_pred) in zip(ejes[1:], resultados.items()):
    eje.scatter(y_test, y_pred, alpha=0.6)
    eje.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="prediccion perfecta")
    eje.set_title(f"{nombre}  (R2 = {r2_score(y_test, y_pred):.3f})")
    eje.set_xlabel("Valor real")
    eje.set_ylabel("Valor predicho")
    eje.legend()

plt.tight_layout()
plt.show()
