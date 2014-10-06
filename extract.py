#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift

from abel import imbin,fourier,invabel
from imbin import imbin2 as imbin

def invfourier(image):
    return ifftshift(np.fft.ifftn(image))

def quantize(x):
    middle = np.median(x)
    mask = x >= middle
    upmean = np.mean(x[mask])
    downmean = np.mean(x[np.logical_not(mask)])
    x[mask] = 1 #upmean
    x[np.logical_not(mask)] = 0 # downmean
    print(upmean)
    print(downmean)
    del mask
    return x

if __name__=="__main__":
    #image = misc.imread("14-076.80000.0V.18500X.Parnell.0001.jpg")
    image = misc.imread("14-076.80000.0V.13000X.Parnell.0002.jpg")
    #image = np.mean(image,axis=2)
    print(image.shape)
    plt.imshow(image)
    plt.show()

    transform = invfourier(image)
    plt.imshow(np.abs(transform))
    plt.show()

    pi = np.pi
    size = image.shape[0]

    q,f = imbin(np.abs(transform))
    a = invabel(f)

    #scale = 1000.0/180.0
    scale = 1000.0/125.0
    
    plt.plot(scale*size/q,f/np.max(f[1:]))
    plt.plot(scale*size/(q+1),a/np.max(a[1:]))
    plt.ylim([-1,1])
    plt.xlim([200,900])
    plt.show()
