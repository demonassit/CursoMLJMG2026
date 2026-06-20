#SNE es una tecnica de reduccion de dimensionalidad especialmente diseñada para VISUALIZAR datos de alta dimension 2d o 3d
# a diferencia de PCA (buscar direccion maxima de la varianza), se enfoca en preservar las relacionas vectoriales locales, osea los puntos mas cercanos a su ortogonalidad
# calculo de probabilidades de que dos vecinos en el espacio ortogonal se correlacionen entre si y crear dimensiones similares o distribuciones similares
# ajustar iterativamente las posiciones de las distribuciones

from sklearn.manifold import TSNE
#vamos a ocupar un dataset de 1797 imaganes con digitos (0-9) de 8x8
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


#este dataset tiene 1797 imaganes de digitos escritos a mano 
digits = load_digits()
X, y = digits.data, digits.target

# crear los modelos de acuerdo al siguiente parametro: 
# tenemos que reducri las dimensiones para poderlo graficar en 2d 
# tenemos que controlar cuantos vecinos se deben de considerar para que cada punto que se va a calcular entre 5 y 50 
# fijar la semilla
# definir el numero de iteraciones para hacerlo lo mas optimo posible
tsne = TSNE(n_components=2, perplexity=30, random_state=40, max_iter=1000)

#vamos a proyecto el resultado
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10,8))

#vamos a graficarlo
scatter = plt.scatter(
    X_tsne[:, 0], #coordenadas 1 en el espacio tsne
    X_tsne[:, 1], #coordenadas 2 en el mismo espacio tsne
    c = y, 
    cmap='tab10', #distignuir los 10 colores de cada digito
    alpha=0.8
)

plt.colorbar(scatter, label='Digito')
plt.title('TSNE De digitos ')
plt.show()
