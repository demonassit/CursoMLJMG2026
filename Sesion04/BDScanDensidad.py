# BDScan definie el radio de vecindad de cada punto llamado epsilon (eps), si dos puntos estana a distancias <= eps son vecinos, un punto con al menos min_samples, vecinos dentro de eps es el nucleo, y los nucleos y sus vecinos forman clusters, los puntos sin ningun nucleo cercano son etiquetados como -1 (ruido)

# los efectos epsilon pueden variar acorde a los dataset
# eps es muy pequeño radio es estrecho, pocos vecinos y muchos puntos son ruido, eso fragmenta el modelo 
# si eps es adecuado significa que el radio es justo, detecta correctamente las regiones densas, y el numero de clusters es coherente a la estructura
# si eps es grande, radio es muy amplio, casi todos los puntos son vecinos entre si, por lo tanto BDScan fusiona todo en un solo cluster

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# vamops a realizar un ejemplo en el cual vamos a tomar dos medias lunas, de forma no convex para verificar que los datos sean realistas 

X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)

# tenemos que medir la distancia euclidiana de los parametros que vamos a obtener, para ello tenemos que aplicarlo con un formato estandar
X = StandardScaler().fit_transform(X)

#configuramos los valores de eps, con el mismo de min_samples, para aislar los efectos
configuraciones = [
    {'eps':0.05, 'titulo':"eps= 0.05 (muy pequeño)\n radio estrecho"},
    {'eps':0.20, 'titulo':"eps= 0.20 (adecuado)\n detectar las 2 lunas de forma correcta"},
    {'eps':0.80, 'titulo':"eps= 0.80 (muy grande)\n todo esta fusionado"}
]

# vamos a entrenarlo
fig, axes = plt.subplots(1,3, figsize=(16,5))
fig.suptitle('BDScan - Efecto del parametro de eps, sobre la densidad de datos \n {dataset make_moons}', fontsize=13, fontweight='bold')

for ax, cfg in zip(axes, configuraciones):
    db = DBSCAN(eps=cfg['eps'], min_samples=5)
    labels = db.fit_predict(X)

    #las metricas
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = (labels == -1).sum()

    print(f"eps = {cfg['eps']:.2f} => Clusters : {n_clusters} | " f"Ruido: {n_ruido} ({n_ruido/len(X)*100:.1f}%)")

    #colorear todos los elementos por cluster
    mascara_validos = labels >= 0
    ax.scatter(
        X[mascara_validos, 0], X[mascara_validos, 1],
        c=labels[mascara_validos],
        cmap='tab10', alpha=0.7, s=15, label=f'{n_clusters} clusters'
    )

    #colorear el ruido
    mascara_ruido = labels == -1
    if mascara_ruido.any():
        ax.scatter(
            X[mascara_ruido, 0], X[mascara_ruido, 1],
            c='black', alpha=0.6, s=25, marker='x',  label=f'{n_ruido} ruido total'
        )
        
    ax.set_title(cfg['titulo'], fontsize=10)
    ax.set_xlabel('Caracteristica 1')
    ax.set_ylabel('Caracteristica 2')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


