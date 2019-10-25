#!/usr/bin/env python
# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## TAREA : Funciones de filtrado en frecuencia
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

# In[36]:



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


# In[40]:


def reimport(module_name: str, alias: str = None, partial: bool = False) -> NoReturn:
    """
        Useless piece of code, literally.
    """
    _exec_str: str = ''
    
    if alias and not partial:
        _exec_str  = f"import {module_name} as {alias}\n"
        _exec_str += f"importlib.reload({alias})\n"
        _exec_str += f"from {module_name} import * \n"
    elif alias and partial:
        _exec_str  = f"import {module_name} as {alias}\n"
        _exec_str += f"importlib.reload({alias})\n"
    elif not alias and not partial:
        _exec_str  = f"import {module_name} as __reimport_tmp\n"
        _exec_str += f"importlib.reload(__reimport_tmp)\n"
        _exec_str += f"from {module_name} import * \n"
        
    exec(_exec_str)

##


# In[48]:


# Importamos todas nuestras funciones (le Gus):
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[3]:


# Importamos todas nuestras funciones (la Pats):
import filtro_notch as lapats
importlib.reload(lapats)
from filtro_notch import *


# In[4]:


def black_and_white(input_image_path):
   return Image.open(input_image_path).convert('L')


# In[5]:


plt.rcParams['figure.figsize'] = (10, 10)


# In[6]:


eps = np.finfo(float).eps
eps.setflags(write=False)


# In[7]:


I = img.imread('imagenes/mama.tif')
plt.imshow(I, cmap='gray')


# In[8]:


fft_viz(I)


# In[9]:


img_surf(I)


# In[10]:


HighI = kernel_highpass(pre_fft_processing(I), Do=1500, form='gauss')


# In[11]:


img_surf(HighI)


# In[12]:


plt.imshow(HighI, cmap='gray')


# In[13]:


list(map(cv2.getOptimalDFTSize, I.shape))


# In[14]:


I.shape


# In[15]:


newI = pre_fft_processing(I)


# In[57]:


x = black_and_white('imagenes/RadiografiaRuidoCoherente.jpg')


# In[58]:


#io.m


# In[59]:


plt.imshow(x, cmap='gray')


# In[19]:


fft_viz(x)


# In[22]:


"""
    Ideas = crear una matriz de desplazamientos.
    
"""


# In[23]:


img_surf(distance_meshgrid_2D(I))


# In[26]:


plt.imshow(I, cmap='gray')


# In[25]:


plt.imshow(kernel_highpass(I, Do=500, form='ideal'), cmap='gray')


# In[56]:


img_surf(kernel_lowpass(I, form='btw', Do=640, n=9), colormap=cm.viridis)


# In[ ]:


img_surf(ker )


# In[40]:


dir(cm)


# In[41]:


type(cm.coolwarm)


# In[29]:


aiuto = aiuda = jelp = help


# In[64]:


aiuda(kernel_ideal)


# In[66]:


kernel_ideal(I.shape[0], I.shape[1], 0, (100, 400), 0, 100)


# In[30]:


aiuto(master_kernel)


# In[44]:


reimport('mfilt_funcs')


# In[55]:


plt.imshow(master_kernel(I, Do=100, kind='high', form='gauss'), cmap='gray')


# In[63]:


fft_viz(x)


# In[ ]:




