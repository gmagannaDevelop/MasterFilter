
from typing import Tuple

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def img_surf(image: np.ndarray) -> None:
    """
    """
    fig  = plt.figure()
    ax   = fig.gca(projection='3d')
    x, y = list(map(lambda x: np.arange(0, x), image.shape))
    X, Y = np.meshgrid(x, y)
    #U, V = fourier_meshgrid(image)
    #print(f'Shapes X:{X.shape}\n Y:{Y.shape}\n Z:{Z.shape}')

    surf = ax.plot_surface(X, Y, image.T, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    
    plt.show()
##

def fourier_meshgrid(image: np.ndarray) -> Tuple[np.ndarray]:
    """
        Genera los arreglos bidimensionales U y V necesarios para poder hacer tanto
        filtrado en frecuencias como la visualización de imágenes en forma de superficies.
        Esto se hace mapeando las intensidades a los valores que tomará la función en el eje
        Z, dados los valores de X y Y que son las coordenadas de los pixeles.
        
    
    Parámetros :
        imagen : Arreglo bidimensional de numpy (numpy.ndarray), es decir una imagen.
        
    Regresa :
        (U, V) : Tuple contieniendo dos arreglos bidimensionales de numpy (numpy.ndarray)
    """
    M, N = image.shape
    u, v = list(map(lambda x: np.arange(0, x), image.shape))
    idx, idy = list(map(lambda x, y: np.nonzero(x > y/2), [u, v], image.shape))
    u[idx] -= M
    v[idy] -= N
    V, U = np.meshgrid(v, u)
    
    return U, V
##
