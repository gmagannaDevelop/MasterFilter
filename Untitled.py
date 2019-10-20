#!/usr/bin/env python
# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## Segundo Examen Parcial
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

# ## Módulos necesarios :

# In[10]:


import numpy as np
import scipy.fftpack as F
import scipy.io as io

import cv2
import matplotlib.image as img

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


# ## Definición de funciones :

# In[43]:


def img_fft(image: np.ndarray, shift: bool = True):
    """
        Ejecutar una Transformada de Fourier adecuada 
        para el procesamiento digital de imágenes.
        
        Parámetros :
                image : Imagen, representada como un arreglo de numpy (numpy.ndarray)
                shift : Booleano que indica si debe ejecutarse la traslación de la imagen e
                        en el espacio de frecuencia.
    """
    _X = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    if shift:
        _X_shift = np.fft.fftshift(_X)
    _X_complex = X_shift[:,:,0] + 1j*X_shift[:,:,1]
    _X_abs = np.abs(_X_complex) + 1 # Evitar que el logaritmo reciba 0 como argumento.
    _X_bounded = 20 * np.log(_X_abs)
    _X_img = 255 * _X_bounded / np.max(_X_bounded)
    _X_img = _X_img.astype(np.uint8)
    
    return _X_img

def ImPotencia(image: np.ndarray):
    """
    """
    return np.sum(np.abs(image)**2) / np.prod(image.shape)


# In[23]:


x = img.imread('docs/Fig.tif')
#print(f'{type(x[0][0])}')
plt.imshow(x, cmap='gray')


# In[35]:


X  = img_fft(x)


# In[36]:


plt.imshow(X, cmap='gray')


# In[44]:


ImPotencia(X)


# In[40]:


np.sum(np.abs(X)**2)


# In[42]:


np.prod(X.shape)


# In[ ]:




