# vamos a buscar en una imagen sus colores y tonalidades, y loq ue vamos hacer es agrupar pixeles para que identifique RGB

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt

# una imagen de color es una muestra en 3d de segmentos de patrones en RGB,  recordemos que son 3 matrices una R, G, B 
# kmeans va agrupar los pixeles y debe de identificarlos por su similitud de su color, para eso cada cluster representa un color promedio, y cada pixel debe de ser reemplazado por su centroide 

img = load_sample_image('flower.jpg')

alto, ancho, canal = img.shape

print(f"Forma original de la imagen {img.shape}")
print(f"Pixeles totales : {alto, ancho}")

# preprocesamiento para la normalización de los kmeans [0,1], dentro del rango de 3 matrices con todas las combinaciones de 0 a 255

img_normalizada = img/255.0

#tenemos que aplanarla altoXanchoX3 para tener las matrices RGB
pixeles = img_normalizada.reshape(-1, 3)

print(f"Formar tras aplanar {pixeles.shape}")

# vamos a segmentar con kmeans los 3 valores de k para compararlos
# k debe de ser al menos 2 hasta 16

valores_k = [2,8,16]
imagenes_segmentadas = []

for k in valores_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(pixeles)

    # a partir del centroide de su cluster tenemos que reemplazar cada pixel, para que se encuentre un valor [0,1]
    colores_centroides = km.cluster_centers_
    pixeles_segmentados = colores_centroides[km.labels_]

    #tenemos que reconstruir la imagen
    img_seg = (pixeles_segmentados.reshape(alto, ancho, 3)*255).astype(np.uint8)
    imagenes_segmentadas.append(img_seg)
    print(f"k = {k:>2} segmentación completa")

fig, axes = plt.subplots(1,4, figsize=(18,5))
fig.suptitle('Segmentación de imagenes con Kmeans \n cada pixel se reemplaza por el color de su centroide', fontsize=13, fontweight='bold')

#imagen original
axes[0].imshow(img)
axes[0].set_title('Imagen Original')
axes[0].axis('off')

#la imagenreconstruida
titulos = [f'k = {k}\{k} Colores unicos' for k in valores_k]

for ax, img_seg, titulo in zip(axes[1:], imagenes_segmentadas, titulos):
    ax.imshow(img_seg)
    ax.set_title(titulos)
    ax.axis('off')

plt.tight_layout()
plt.show() 
