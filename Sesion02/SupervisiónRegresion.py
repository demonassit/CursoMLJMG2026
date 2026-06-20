#vamos con un flujo universal de regresion lineal
# Vamos a cargar unos datos y esos datos los vamos a dividir en elementos para el entrenamiento y elementos para la prueba
# vamos a entrenar el modelo 
# vamos a predecir su comportamiento

from sklearn.datasets import fetch_california_housing
#es un dataset de casas de california
from sklearn.linear_model import LinearRegression
#vamos a necesita tomar secciones definidas para el entrenamiento
from sklearn.model_selection import train_test_split
#necesitamos realizar el calculo de MSE 
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

# este dataset contiene 20640 muestras de casitas con 8 caracteristicas 
# nuestras variables objetivo son y = precio mediano de la casa por cada 100 000 dolares

housing = fetch_california_housing()

X, y = housing.data, housing.target # x es todo el dataset y el precio

# De todo este dataset tenemos que definri cuantos datos son para el entrenamiento y cuantos son para la prueba, 80 20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=40
)

#definimos el modelo

modelo = LinearRegression()

modelo.fit(X_train, y_train) # las que van a entrenar al modelo
#necesitamos la predicción
y_pred = modelo.predict(X_test)


#la regresion lineal es el calculo de los minimos cuadrados, significa que un coeficiente de su determinación no de presición, por lo tanto r2 = 1.0, y la predicción es perfecta si r2=0.0, esto equivale a decir siempre la media de y va a estar entre los valores de 0.1 - 0.0 son valores superiores al .8 valores .9 tenemos mucha presición

r2 = modelo.score(X_test, y_test)

#aplicamos el calculo de los errores respecto de sus valores

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 = {r2: .4f}")
print(f"RMSE = {rmse: .4f} en $100,000.00 USD")

#vamos a pintarlo
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
minimo, maximo = y_test.min(), y_test.max()
plt.plot([minimo, maximo], [minimo, maximo], 'r--', linewidth=1.5, label='Predicción por LR')
plt.xlabel('Precio real ($100, 000.00 USD)')
plt.ylabel('Precio predicho respecto al original')
plt.title('Regresion Lineal de Casitas en California')
plt.legend()
plt.tight_layout()
plt.show()

