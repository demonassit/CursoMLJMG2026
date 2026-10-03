# En este ejercicio comparamos tres formas de combinar modelos (ensambles) sobre el mismo dataset de Breast Cancer
# que ya usamos con Random Forest, asi podemos ver cual mejora y por que.
#
# 1. Bagging (Random Forest): entrena muchos arboles EN PARALELO, cada uno con una muestra aleatoria de los datos,
#    y al final votan. Reduce la varianza (el sobreajuste de un arbol solo).
# 2. Boosting (AdaBoost): entrena arboles muy simples EN SECUENCIA. Cada arbol nuevo le da mas peso
#    a los ejemplos que el anterior clasifico mal, asi el modelo va corrigiendo sus errores paso a paso.
# 3. Stacking: combina modelos DIFERENTES (RF, KNN, SVM). Sus predicciones se convierten en las entradas
#    de un modelo final (meta-modelo) que aprende en cual confiar mas.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# el pipeline une el escalado con el modelo, asi el escalado se aprende solo con los datos de entrenamiento

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report

import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target  # 0 maligno 1 benigno

# misma division que en Supervisado.py: 75% entrenamiento, 25% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=50, stratify=y
)

# ---------------- Definimos los modelos ----------------

# modelo base: un solo arbol sin limites (tiende a sobreajustar)
arbol = DecisionTreeClassifier(random_state=50)

# bagging: el Random Forest de la clase anterior
rf = RandomForestClassifier(n_estimators=200, random_state=50)

# boosting: AdaBoost con "tocones" (stumps), arboles de profundidad 1 que solo hacen UNA pregunta
# n_estimators = cuantos tocones entrenar en secuencia
# learning_rate = que tanto pesa cada tocon nuevo (mas bajo = aprende mas lento pero mas estable)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=300,
    learning_rate=0.5,
    random_state=50
)

# stacking: tres modelos distintos + un meta-modelo
# KNN y SVM miden distancias, por eso llevan escalado; los arboles no lo necesitan
modelos_base = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=50)),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, random_state=50))),
]
# cv=5: las predicciones que recibe el meta-modelo salen de validacion cruzada,
# asi no aprende de predicciones "tramposas" sobre datos que los modelos base ya vieron
stacking = StackingClassifier(
    estimators=modelos_base,
    final_estimator=LogisticRegression(),
    cv=5
)

modelos = {
    'Arbol de decision': arbol,
    'Random Forest (bagging)': rf,
    'AdaBoost (boosting)': ada,
    'Stacking': stacking,
}

# ---------------- Validacion cruzada y prueba ----------------
# la validacion cruzada (5 partes) nos dice que tan estable es cada modelo, no solo si tuvo suerte con un split
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

resultados = {}
print(f"{'Modelo':<26}{'CV accuracy':>18}{'Test acc':>10}{'F1':>8}{'AUC':>8}")
for nombre, modelo in modelos.items():
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    resultados[nombre] = {
        'cv_media': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc(fpr, tpr),
    }
    print(f"{nombre:<26}{cv_scores.mean():>10.4f} +/- {cv_scores.std():.4f}"
          f"{accuracy_score(y_test, y_pred):>10.4f}{f1_score(y_test, y_pred):>8.4f}{resultados[nombre]['auc']:>8.4f}")

print('\nReporte Clasificador del Stacking')
print(classification_report(y_test, stacking.predict(X_test), target_names=['Maligno', 'Benigno']))

# que tanto confia el meta-modelo en cada modelo base (coeficientes de la regresion logistica)
print('Peso que el meta-modelo le da a cada modelo base:')
for (nombre, _), peso in zip(modelos_base, stacking.final_estimator_.coef_[0]):
    print(f"  {nombre:<5} {peso:>6.2f}")

# AdaBoost permite ver como mejora conforme agrega tocones: staged_predict da la prediccion despues de cada paso
acc_por_paso = [accuracy_score(y_test, y_p) for y_p in ada.staged_predict(X_test)]

# ---------------- Graficas ----------------
colores = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # un color fijo por modelo en las tres graficas
nombres = list(modelos.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Ensambles sobre Breast Cancer: bagging vs boosting vs stacking', fontsize=12, fontweight='bold')

# grafica 1: accuracy promedio de la validacion cruzada con su desviacion (barra de error)
medias = [resultados[n]['cv_media'] for n in nombres]
stds = [resultados[n]['cv_std'] for n in nombres]
axes[0].barh(nombres, medias, xerr=stds, color=colores, capsize=4)
for i, m in enumerate(medias):
    axes[0].text(0.855, i, f'{m:.3f}', va='center', ha='left', color='white', fontweight='bold')
axes[0].set_xlim(0.85, 1.02)
axes[0].invert_yaxis()
axes[0].set_xlabel('Accuracy (validacion cruzada, 5 partes)')
axes[0].set_title('Desempeño promedio y variacion')
axes[0].grid(axis='x', alpha=0.2)

# grafica 2: curvas ROC de los cuatro modelos sobre el conjunto de prueba
for nombre, color in zip(nombres, colores):
    r = resultados[nombre]
    axes[1].plot(r['fpr'], r['tpr'], color=color, linewidth=2, label=f"{nombre} (AUC = {r['auc']:.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador Aleatorio')
axes[1].set_xlabel('FPR tasa de Falsos Positivos')
axes[1].set_ylabel('TPR tasa de Verdaderos Positivos')
axes[1].set_title('Curva ROC')
axes[1].legend(loc='lower right', fontsize=8)
axes[1].grid(alpha=0.2)

# grafica 3: AdaBoost aprendiendo paso a paso
axes[2].plot(range(1, len(acc_por_paso) + 1), acc_por_paso, color=colores[2], linewidth=2)
axes[2].set_xlabel('Numero de tocones (arboles de profundidad 1)')
axes[2].set_ylabel('Accuracy en prueba')
axes[2].set_title('AdaBoost: cada tocon corrige errores del anterior')
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.show()
