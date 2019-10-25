#!/usr/bin/env python
# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## TAREA : Funciones de filtrado en frecuencia
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

# In[1]:



import copy
import importlib
from typing import Tuple, List, NoReturn

import numpy as np
import scipy.fftpack as F
import scipy.io as io

import cv2
import matplotlib.image as img

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import skimage
import skimage.morphology
import skimage.filters

from PIL import Image

import scipy.io as io


# In[54]:


# Importamos todas nuestras funciones:
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[59]:


def plot_all(image, **kw): 
    """
    """
    
    fig = plt.figure(figsize = (15, 10))
    H = master_kernel(image, **kw)
    filtrada = filtra_maestra(image, **kw)
    
    fig.add_subplot(2, 3, 1)
    plt.imshow(image, cmap = 'gray')
    plt.title('Imagen original', size = 18)
    
    fig.add_subplot(2, 3, 2)
    fft_viz(image)
    plt.title('Transformada de la imagen', size = 18)
    
    fig.add_subplot(2, 3, 3)
    plt.imshow(filtrada, cmap = 'gray')
    plt.title(f'Imagen filtrada.', size = 18)


# In[74]:


plt.rcParams['figure.figsize'] = (10, 10)


# In[75]:


eps = np.finfo(float).eps
eps.setflags(write=False)


# In[76]:


I = img.imread('imagenes/mama.tif')
plt.imshow(I, cmap='gray')


# In[77]:


fft_viz(I)


# In[78]:


img_surf(I)


# In[79]:


x = cv2.imread('imagenes/RadiografiaRuidoCoherente.jpg', 0)


# In[80]:


plt.imshow(x, cmap='gray')


# In[81]:


fft_viz(x)


# In[11]:


img_surf(x)


# ### 5.1 Filtro pasa bajos ideal con wc=64,

# In[67]:


banderas = dict(Do=64, kind='lowpass', form='ideal')
plot_all(I, **banderas)


# ### 5.2 Filtro pasa bajos butt con wc=64, orden=2

# In[66]:


banderas = dict(Do=64, kind='lowpass', form='btw', n=2)
plot_all(I, **banderas)


# ### 5.3 Filtro pasa bajos gauss con wc=64

# In[65]:


banderas = dict(Do=64, kind='lowpass', form='gauss')
plot_all(I, **banderas)


# ### 5.4 Filtro pasa altos gauss con wc=64

# In[63]:


banderas = dict(Do=64, kind='highpass', form='gauss')
plot_all(I, **banderas)


# ### 5.5 Filtro pasa bandas gauss con wc1=54, wc2=74

# In[62]:


banderas = dict(wc1=54, wc2=74, kind='bandpass', form='gauss')
plot_all(I, **banderas)


# ### 5.6 Filtro rechazo de bandas gauss con wc1=54, wc2=74

# In[61]:


banderas = dict(wc1=54, wc2=74, kind='bandreject', form='gauss')
plot_all(I, **banderas)


# In[82]:


cabeza = plt.imread('imagenes/FigP0405(HeadCT_corrupted).tif')


# ### Notch reject, Do = 5, center = (0, 20), form='ideal'

# In[86]:


banderas = dict(kind='notchreject', Do=5, center=(0, 20))
plot_all(cabeza, **banderas)


# In[88]:


banderas = dict(kind='notchreject', Do=5, center=(11, 0), form='gauss')
plot_all(cabeza, **banderas)


# In[73]:


img_surf(master_kernel(I, kind='bandreject', form='btw', wc1=100, wc2=250, n=2))


# In[ ]:





# In[ ]:





# In[ ]:




