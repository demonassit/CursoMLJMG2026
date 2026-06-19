# vamos a realizar un ejemplo donde tenemos que aplicar PCA a una matriz utilizando elementos de algebra lineal SVD a traves de la libreria de Numpy para poder compara los resultados de un caso practico

import numpy as np
import matplotlib.pyplot as plt
# vamos a ocupar de la libreria sklearn un elemento para este ejemplo

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

#aqui vamos a tener un dataset de imagenes de muestras de tumores malignos como benignos 0 y 1

#vamos a cargar el dataset en nuestra variable
data = load_breast_cancer() #este tiene todos los elementos con sus dimensiones del dataset
X = data.data #matriz de todas las caracteristicas
y = data.target #etiquetando para cada clase

#tenemos que estandarizar las variables para reducir las dimensiones
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#vamos aplicar los elementos del calculo
# V = vectores singulares izquierdos de la matriz de 3 variables
# S = valores singulares que se van a obtener referente de la magnitud de cada componente
# Vh = los vectores singulares derechos a partir de la matriz transpuesta 

V, S, Vh = np.linalg.svd(X_scaled, full_matrices=False)

#vamos a extraer cada componente
pc1 = Vh[0] #nuestro primer componente principal le tenemos que decir cuantas variables necesitamos
pc2 = Vh[1] #el valor ortogonal respecto de pc1

#de todo este calculo ya tenemos la base para la identificación

W = Vh[:2].T # aqui estamos formando nuestras variables componentes para extraer los coeficientes de la matriz transpuesta

#aplicamos la varianza
varianza_total = (S**2).sum()
varianza_explicada = (S[:2]**2)/varianza_total

print(f"Varianza explicada por cada componente: {varianza_explicada}")
print(f"Varianza total retenida: {varianza_total}")

plt.figure(figsize=(8,6))

plt.scatter(
    X_nueva[:, 0], #componente principal para x
    X_nueva[:, 1], #componente principal para y
    c = y, # segun la clase malingo [0] y benigno[1]
    cmap = 'coolwarm', # paleta de colores
    aplha = 0.7 # es el nivel de transparencia de los puntos que se superponen  
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA con Numpy para detección de cancer')
plt.colorbar(label='Clase')
plt.show()


