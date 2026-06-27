# k means es un algoritmo de aprendizaje no supervisado, de agrupamiento, (clusters) a diferencia de todo lo anterior aqui no hay etiquetas. 
# 1.- elegimos k puntos aleatorios como centroides iniciales
# 2.- asignar cada punto al centroide mas cercano calculo euclidiano
# 3.- recalculamos el centroide de cada grupo como el promedio de sus puntos
# 4.- repetimos los pasos 2 y 3 hasta que los centroides no se muevan

#los metodos del codo grafica la inercia (suma de distancias al cuadrado de cada punto del centroide), la distancia mas corta a medida que k crece la inercia baja. El codo de la curva (donde baja) indica el k optimo ojo añadir mas cluster no mejora la conexion

#coeficiente de la silueta mide que tan bien definido esta cada cluster, compara la distancia media de un punto a los demas de su propio cluster, (cohesion), contra la distancia media al cluster vecino ms cercano

#ejericicio vamos a generar muchos datos artificiales, con 4 grupos conocidos para aplicar k means para valores de k entre 2 y 9 y los vamos a graficar

from sklearn.cluster import KMeans

#vamos a definir los elementos de las metricas de evaluación de KMeans
from sklearn.metrics import silhouette_score # cohesion y separación

#vamos a generar los datos artificiales
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# de esos datos make_blobs vamos a generar puntos aleatorios con 500 muestras, 4 centroides y una semilla 

# las variables como X que seran las coordenadas (x,y) para los elementos muestra
# _ como etiquetas reales, de cada punto

X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# tenemos que evaluar las distancias entre cada elemento para su cohesión y separación

inercias = [] #aplicar el metodo del codo
siluetas = [] #aplicar silueta

# probar k desde 2 hasta 9 (kmeans requiere al menos 2 cluster)

# primero creamos el modelo de lso cluster, segundo los agrupamos, y apartir de la semilla que definimos los vamos a grupar, y esto se tiene que ejecutar un numero de veces tal que kmeans se ejecuta hacia el calculo de sus centroides 

k_range = range(2,10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)

    #vamos a entrenar el modelo y devolver las etiquetas que encuentra cada cluster
    labels = km.fit_predict(X)

    inercias.append(km.inertia_)

    siluetas.append(silhouette_score(X, labels))

#graficamos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#puntos azules unidos por una linea
ax1.plot(k_range, inercias, 'bo-', linewidth=2, markersize=7)
ax1.set_title('Metodo del codo')
ax1.set_xlabel('Numero de Clusters')
ax1.set_ylabel('Inercia (Suma de las distancias)')
# cuadrados rojos unidos por una linea
ax2.plot(k_range, siluetas, 'rs-', linewidth=2, markersize=7)
ax2.set_title('Coeficientes de Silueta')
ax2.set_xlabel('Numero de Clusters (k)')
ax2.set_ylabel('Silueta promedio')

plt.suptitle('Seleccion del numero optimo de clusters', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()