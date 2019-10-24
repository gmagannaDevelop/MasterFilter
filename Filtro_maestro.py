#!/usr/bin/env python
# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## TAREA : Funciones de filtrado en frecuencia
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

# In[5]:



import copy
import importlib
from typing import Tuple, List

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


# In[6]:


# Importamos todas nuestras funciones :
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[7]:


plt.rcParams['figure.figsize'] = (5, 5)


# In[8]:


eps = np.finfo(float).eps
eps.setflags(write=False)


# In[9]:


I = img.imread('imagenes/mama.tif')
plt.imshow(I, cmap='gray')


# In[10]:


fft_viz(I)


# In[ ]:




