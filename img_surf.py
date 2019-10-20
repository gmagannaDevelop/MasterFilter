#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.fftpack as F
import scipy.io as io

import cv2 as cv
import matplotlib.image as img

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter


mama = img.imread('imagenes/Mamografia.tif')


fig1 = plt.figure()
plt.imshow(mama, cmap='gray')
plt.show()

mama_f = F.fft(mama)

fig = plt.figure()
ax = fig.gca(projection='3d')
"""
x = np.arange(0, mama.shape[0])
y = np.arange(0, mama.shape[1])
"""
x, y = list(map(lambda x: np.arange(0, x), mama.shape))
X, Y = np.meshgrid(x, y)
Z = mama.T
print(f'Shapes X:{X.shape}\n Y:{Y.shape}\n Z:{Z.shape}')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

plt.show()

