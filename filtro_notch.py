#!/usr/bin/env python
# coding: utf-8

# 2. (3 puntos) Hacer una función que diseñe filtros notch ideales, de Gauss o de Butterworth.

# In[7]:


import numpy as np
import importlib


# In[8]:


# Importamos todas nuestras funciones (le Gus):
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[9]:


def kernel_ideal(M, N, pasa, centro, radio, d0):
    u_k = centro[0]
    v_k = centro[1]
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(u, v)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = D_k = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = np.where(D_k < d0, 0, 1) # Primer pasaaltos
    H_mk = np.where(D_mk < d0, 0, 1) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel


# In[5]:


def filtrar_notch(img, tipo = 0, pasa = 0, centro = (0, 0), radio = 1, d0 = 50, n = 0.0):
    """Filtro notch. 
    tipo = 0 para ideal, 1 para gaussiano y cualquier otro valor para butterworth.
    pasa = 0 para notchreject, 1 para notchpass.
    centro y radio son los del notch. notch simétrico automático.
    Especificar n solo para butterworth"""
    
    M, N = img.shape
    
    if tipo == 0:
        kernel = kernel_ideal(M, N, pasa, centro, radio, d0)
    elif tipo == 1:
        kernel = kernel_gaussiano(U, V, pasa, centro, radio, d0)
    else:
        kernel = kernel_butterworth(U, V, pasa, centro, radio, d0, n)
        
    transformada = np.fft.fftshift(np.fft.fft2(img))
    aplico_filtro = kernel * transformada
    img_filtrada = np.real(np.fft.ifft2(np.fft.ifftshift(aplico_filtro)))
    
    return img_filtrada


# In[ ]:




