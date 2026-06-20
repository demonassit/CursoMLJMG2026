# los clasificadores por KNN y SVM miden distancias euclidianas entre cada uno de los puntos del clasificador, este tipo de clasificador domina el calculo y sesga las predicciones 0 y 1, media es el valor de 0 y 1 de acuerdo a su cercania

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #manejador de escalas
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

#vamos a crear un archivo para comparar florecitas wiiii con sus sepalos y petalos, 150 flores, 3 especies de flores diferentes, con 4 caracteristicas sepalos y petalos
#clases 0 setosa, 1 versicolor 2 virginica

iris = load_iris()
X, y = iris.data, iris.target

#divir el modelo del entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=60, stratify=y #garantizar que las clases se puedan representar
)

#tenemos que ajustar los elementos para el modelo
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) #ajustar t transforma el entrenamiento (matriz transpuesta)
X_test_s = scaler.fit_transform(X_test) # transofrmar los elementos de prueba sin un reajuste

#vamos a comparar que pasa con una regresion lineal con elmentos categoricos
#lo comparamos con KVecinos
#SVM que es para encontrar hiperplanos

modelos = {
    'Reg. Lineal: ': LogisticRegression(max_iter=200),
    'KNN (k=5): ': KNeighborsClassifier(n_neighbors=5),
    'SVM: ' : SVC(kernel='rbf')

}

for nombre, m in modelos.items():
    m.fit(X_train_s, y_train)
    print(f"\n == {nombre} == ")
    print(classification_report(
        y_test, m.predict(X_test_s),
        target_names=iris.target_names
    ))

#necesitamos la matriz de confusion para cuantas muestras vamos a necesitar comparar
fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle('Matrices de Confusion por Clasificador Iris', fontsize=14, fontweight='bold')

for ax, (nombre, m) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, m.predict(X_test_s))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(nombre)
    ax.tick_params(axis='x', labelrotation=20)

plt.tight_layout()
plt.show()