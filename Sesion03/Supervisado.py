# En este ejercicio vamos a utilizar las tecnicas mas importantes para evaluar y optimizar un modelo  de clasificación rigurosa.
# GridSeachCV es para la busqueda de hiperparametros, los cuales son configuraciones que no se aprenden durante el entrenamiento, esos los elige el programador, por ejemplo, el numero arboles, la profundidad del arbol, las combinaciones posibles y la selección del desempeño.

#Validación cruzada, en lugar de evaluar el modelo, con una sola visión de entrenamiento, (train/test) la validacion cruzada divide los datos en k= 5 partes, el modelo se entrena k veces, cada vez que usa k-1 de las partes a entrenar y validar. El desempeño final es el promedio de las evaluaciones.

# Curvas ROC y AUC, para evaluar los umbrales, un clasificador binario. Porque predice la probabilidad del umbral, (0,1) con esta curva nosotros podemos aplicar la matriz de confusión TPR o FPR

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV
# cros va a aevaluar un modelo con multiples cruces
# grid ese va a realizar la busqueda de los hiperparametros a partir de la validación cruzada

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

X, y = data.data, data.target  #0 maligno 1 benigno

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=40, stratify=y
)

#Primero necesitamos es el escalado para mejorar el mejor del mejor del entrenamiento 
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

#para la busqueda de los hiperparametros tenemos que definir sus combinaciones 
# ocupar n_estimator, numero de arboles en el bosque 50 a 100
# debemos definri la profundidad del arbol, Node -> Node   Node -> None, (no tiene limites), 5 a 10
# tenemos que considerar las k (las partes del entrenamiento con sus combinaciones por parte de la validación cruzada)

param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [None, 5, 10]
}

#para los elementos de la busqueda del grid, tenemos que tener un modelo base que debe optimizarse
# estimador = modelo base
# param_grid = son las combinaciones de los hiperparametros
# cv = 5 = validaciones cruzadas de la división de k (significa dividir X_train en 5 partes)
# scoring = 'criterio', f1 es el score que sirve para maximizar F1-Score, (es un clasificador de balance de presición con regresión)

gs = GridSearchCV(
    RandomForestClassifier(random_state=40),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

#ahora si quiere que lo entrenes y loe values tenemos que definir las combinaciones k=5 * n = 30(caracteristicas)
gs.fit(X_train, y_train)

#al modelo ya entrenado vamos con los mejores hiperparametros, para hacer un reentrenamiento
best = gs.best_estimator_

#hay existir un reentrenamiento podemos obtener al mejor modelo, predicción y probabilidad
mejor = gs.best_estimator_
y_pred = mejor.predict(X_test_s)
y_proba = mejor.predict_proba(X_test_s)[:, 1]

print('Reporte Clasificador')
print(classification_report(
    y_test, y_pred, target_names=['Maligno', 'Benigno']
))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Evaluación del mejor modelo de Random Forest\n {gs.best_params_}', fontsize=12, fontweight='bold')

#debemos graficar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Maligno', 'Benigno'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Matriz de Confusión')


#prepara las curvas roc 
fpr, tpr, _ = roc_curve(y_test, y_proba)

# definimos las curvas AUC
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
axes[1].plot([0,1], [0,1], 'k--', linewidth=1, label='Clasificador Aleatorio')
axes[1].set_xlabel('FPR tasa de Falsos Postivos')
axes[1].set_ylabel('FPR tasa de Verdaderos Postivos')
axes[1].set_title('Curva ROC')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.show()





