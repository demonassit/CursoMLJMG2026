#queremos predecir el precio bajo de 3 modelos de entrenamienot de regresion lineal para minimizar el error atipico entre cada predicción

from sklearn.datasets import fetch_california_housing
#es un dataset de casas de california
from sklearn.linear_model import LinearRegression, Ridge, Lasso

#ridge nos sirve para evitar que los coeficientes crezcan demasiado (overfitting) cuando tenemos demasiadas variables

#lasso maneja los valores absolutos, y esto lleva que muchos coeficientes se vayan a cero, esto hace que su seleccion sea automatica para la variable


#vamos a necesita tomar secciones definidas para el entrenamiento
from sklearn.model_selection import train_test_split
#necesitamos realizar el calculo de MSE 
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt


housing = fetch_california_housing()  #MedInc, Age, Rooms, Hab, Poblacion, Lat, Long, Precio

X, y = housing.data, housing.target 

#entrenamiento
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42
)

#vamos a crear una funcion que se encargue de generar dos graficos

def grafica_modelo(y_real, y_pred, nombre, rmse, r2, ax_scatter, ax_error):
    #Generar dos graficos para evaluar el modelo de regresion 
    # 1. Dispersion de los valores reales vs predictivos, comparacipon de sus representaciones mas cercanas entre cada punto respecto de la diagonal 
    # 2. distribucion de errores dentro de la preccion con el fin de evitar algo que este fuera de los parametros de busqueda
    #valores reales vs prediccion
    ax_scatter.scatter(
        y_real, y_pred,
        alpha=0.3,   #esto es para poder transparentar laz zonas de mayor densidad o de mas colisiones
        s=10 #para no saturar la grafica de dispersión
    )

    #las condiciones para la  predccion
    minimo = min(y_real.min(), y_pred.min())
    maximo = max(y_real.max(), y_pred.max())

    ax_scatter.plot(
        [minimo, maximo], [minimo, maximo], 'r--', linewidth=1.5, label='Prediccion'
    )

    ax_scatter.set_xlabel('Precio Real 100K')
    ax_scatter.set_ylabel('Precio Predicho a los 100K')
    ax_scatter.set_title('f{nombre}\n RMSE = {rmse: .4f} R2 = {r2: .4f}')
    ax_scatter.legend(fontsize=8)

    #histograma 
    residuos = y_real - y_pred

    ax_error.hist(
        residuos,
        bins=50, #tamano de la grafica de barra
        color='salmon',
        edgecolor='white',
        linewidth=0.4

    )

    ax_error.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Error = 0')
    ax_error.set_xlabel('Error real - predictivo')
    ax_error.set_ylabel('Frecuencia')
    ax_error.set_title('Distribución de Errores - {nombre}')
    ax_error.legend(fontsize=8)

#ahora tenemos que para el modelo de entrenamiento necesitamos pasar por esas 3 lineas RL, ridge, plasso

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(14, 12)
)

fig.suptitle(
    'Comparacion de modelos de regresion para el caso de las casitas lindas y kawaii',
    fontsize=14, fontweight='bold'
)

#empezamos con el modelo a partir de una matriz para cada elemento
for fila, (nombre, modelo) in enumerate([
    ('Lineal', LinearRegression()),
    ('Ridge', Ridge(alpha=1.0)),
    ('Lasso', Lasso(alpha=0.1))
]):
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # proporcion de la varianza por el modelo de reduccion de coeficientes por el valor absoluto
    # varianza = 0.6 respecto del modelo 
    r2 = r2_score(y_test, y_pred)

    print(f"{nombre}: RMSE={rmse: .4f}, R2={r2: .4f}")

    grafica_modelo(
        y_test, y_pred, nombre, rmse, r2,
        ax_scatter=axes[fila, 0],
        ax_error=axes[fila, 1]
    )

#metemos un ajuste a las graficas
plt.tight_layout()
plt.show()
