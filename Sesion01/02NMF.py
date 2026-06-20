# Vamos a cargar un dataset de rostros humanos, alrededor de 400 fotografias las cuales vamos a reconocer, de ese dataset tomaremos 40 elementos para poder aplicar una matriz de factorización no negativa, esto es una tecnica de reduccion de variables o dimensiones, a partir de una matriz X con dos matrices W y H ambas con valores no negativos

# x = W * H
# datos = pesos * componentes
# los pesos son las caracteristicas que nosotros bucamos por ejemplo rasgos faciales, caracteristicas unicas del rostro, etc
# los componentes son los elementos de clasificación para cada rostro por ejemplo, ojos cafes, ojos azules, nariz chata, etc.

from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

# este dataset tiene 400 rostros, de 40 personas diferentes, y necesitamos al menos 10 fotos de cada persona, que se a representar por un vector de tamaño 4096 64x64 por lo que su tamaño es de 400, 4096
# para este tratamiento lo haremos a partir de elementos aleatorios y debemos obtener una semilla

faces = fetch_olivetti_faces(shuffle=True, random_state=40) 

#vamos a necesitar nuestra matriz
X = faces.data 

# vamos a crear nuestro modelos para identificar esos rostros por ejemplo 15 rostros
nmf = NMF(n_components=15, random_state=60)

#vamos ajustar el modelo, para obtener los "pesos" 
#primero debemos obtener los rostros
#tenemos que darle la forma (400, 15)
# cada imagen la debemos reconstruir (cada pixel)

X_nmf = nmf.fit_transform(X)

fig, axes = plt.subplots(3,5, figsize=(12,8))

# vamos a recorrer cada subgrafia, y dibujamos sus componentes

for i, ax in enumerate(axes.ravel()):
    #ravel() convirtiendolo en una cuadricula de la lista plana
    #de cada componente[i] va a tener un vector de tamaño 4096
    #lo vamos a redimensionar (la reconstrucción) a 64X64
    ax.imshow(nmf.components_[i].reshape(64,64), cmap='gray')
    #ocultamos los ejes para visualizarlo
    ax.axis('off')

plt.suptitle('Componentes de NMF de Rostros')
plt.show()
