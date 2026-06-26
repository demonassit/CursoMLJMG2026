# un arbol de desiciones es modelo que aprende una serie de condiciones sobre caracteristicas, organizadas de forma de arbol para llegar a una prediccion
# El problema de los arboles de desición es el overfitting (sobreajuste), memorizar loss datos de entrenamiento tan bien que pierde la capacidad de generar datos nuevos

#vamos a realizar un ejemplo sobre una muestra aleatoria de un dataset y un subconjuto aleatorio de caracteriscas, en el cual vamos a generar 100 arboles usando Breast Cancer 

from sklearn.ensemble import RandomForestClassifier
#vamos a comprar arboles de desición con el RF
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # propocion de prediccion correcta a partir de errores

import numpy as np
import matplotlib.pyplot as plt

#este dataset tiene 568 muestras de tumores de cancer con 30 caracteristicas a partir de biopsias malignos y benignos

data = load_breast_cancer()
X, y = data.data, data.target # el diagnostico 

#entrenamiento 75 25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=40
)

#la creacion del modelo, que tiene una caracteristica sin restricciones, crezca hasta que clasifique perfectamente los datos 
dt = DecisionTreeClassifier(random_state=40).fit(X_train, y_train)

#otro arbol de tipo random forest, donde nosotros vamos a limitar su crecimiento con 100 arboles, y debemos establecer el numero de iteraciones
rf = RandomForestClassifier(n_estimators=100, random_state=40).fit(X_train, y_train)

#calcular el nivel de predicción de cada arbol
print(f'Arbol de desiciones: {accuracy_score(y_test, dt.predict(X_test)):.4f}')
print(f'Random Forest: {accuracy_score(y_test, rf.predict(X_test)):.4f}')

#la importancia de las caracteristicas del atributo de random forest que indica cuanto contribuyo cada caracteristica a mejora dentro de las pruebas de cada nodo
importancia = rf.feature_importances_

#los elementos del ordenamiento de los indices de menor a mayor importancia
# [ : : -1  ] los invierte de mayor a menor
# [ : 10  ] tomar solo los 10 mas importantes
indices = np.argsort(importancia)[::-1][:10]

#graficamos
plt.barh(
    range(10),
    importancia[indices],
    color='steelblue', edgecolor='white'
)

#vamos a obtener los nombres de las caracteristicas del dataset
plt.yticks(range(10), [data.feature_names[i] for i in indices])

plt.xlabel('Importancia (reduccion media del indice: )')
plt.title('Top 10 de caracteristicas importantes con RF')
#invertir el orden de las caracteristicas
plt.gca().invert_yaxis
plt.show()