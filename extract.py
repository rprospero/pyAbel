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

def extract(imageFile,bar,pixels):
    image = misc.imread(imageFile)
    print(image.shape)
    #plt.imshow(image)
    #plt.show()

    transform = invfourier(image)
    #plt.imshow(np.abs(transform))
    #plt.show()

    pi = np.pi
    size = image.shape[0]

    q,f = imbin(np.abs(transform))
    a = invabel(f)

    scale = bar/pixels

    print(scale)
    print(size)
    print(scale*size)
    
    #plt.plot(scale*size/q,f/np.max(f[1:]))
    wave = scale*size/(q+1)
    mask = np.logical_and(wave > 200, wave < 1000)

    return (wave[mask],a[mask])


if __name__=="__main__":
    data = [("14-076.80000.0V.13000X.Parnell.0002.jpg",1000,125,"2"),
            ("14-076.80000.0V.13000X.Parnell.0003.jpg",1000,125,"3"),
            ("14-076.80000.0V.13000X.Parnell.0005.jpg",1000,125,"5"),
            ("14-076.80000.0V.13000X.Parnell.0007.jpg",1000,125,"7"),
            ("14-076.80000.0V.18500X.Parnell.0001.jpg",1000,180,"1"),
            ("14-076.80000.0V.18500X.Parnell.0004.jpg",1000,180,"4"),
            ("14-076.80000.0V.18500X.Parnell.0006.jpg",1000,180,"6"),
            ("14-076.80000.0V.18500X.Parnell.0008.jpg",1000,180,"8")]
    for f,b,p,t in data:
        q,a = extract(f,b,p)
        plt.plot(q,a,label=t)
    plt.legend()
    plt.show()
