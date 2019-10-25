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

from PIL import Image

import scipy.io as io


# In[42]:


# Importamos todas nuestras funciones :
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[3]:


def black_and_white(input_image_path):
   return Image.open(input_image_path).convert('L')


# In[4]:


plt.rcParams['figure.figsize'] = (10, 10)


# In[5]:


eps = np.finfo(float).eps
eps.setflags(write=False)


# In[6]:


I = img.imread('imagenes/mama.tif')
plt.imshow(I, cmap='gray')


# In[7]:


fft_viz(I)


# In[18]:


img_surf(I)


# In[16]:


HighI = kernel_highpass(pre_fft_processing(I), Do=1500, form='gauss')


# In[17]:


img_surf(HighI)


# In[85]:


plt.imshow(HighI, cmap='gray')


# In[60]:


list(map(cv2.getOptimalDFTSize, I.shape))


# In[62]:


I.shape


# In[19]:


newI = pre_fft_processing(I)


# In[20]:


x = black_and_white('imagenes/RadiografiaRuidoCoherente.jpg')


# In[21]:


#io.m


# In[22]:


plt.imshow(x, cmap='gray')


# In[23]:


fft_viz(x)


# In[89]:


x.shape


# In[24]:


#help(plt.imread)


# In[25]:


"""
    Ideas = crear una matriz de desplazamientos.
    
"""
I


# In[26]:


U, V = fourier_meshgrid(I)
D = fourier_distance(U, V)
H = np.zeros_like(D)


# In[27]:


dd = (D / D.max())*255


# In[28]:


di = np.uint8(dd)


# In[29]:


img_surf(distance_meshgrid_2D(I))


# In[30]:


plt.imshow(kernel_band_pass(I, wc1=201, wc2=500, form='btw'), cmap='gray')


# In[56]:


img_surf(kernel_lowpass(I, form='btw', Do=640, n=9), colormap=cm.viridis)


# In[ ]:


img_surf(ker )


# In[40]:


dir(cm)


# In[41]:


type(cm.coolwarm)


# In[ ]:




