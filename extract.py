#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift

from abel import imbin,fourier,invabel
from imbin import imbin2 as imbin

import sqlite3

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

    transform = invfourier(image)

    pi = np.pi
    size = image.shape[0]

    q,f = imbin(np.abs(transform))
    a = invabel(f)

    print(bar)
    print(pixels)
    scale = bar/pixels

    print(scale)
    print(size)
    print(scale*size)
    
    wave = scale*size/(q+1)
    mask = np.logical_and(wave > 200, wave < 1000)

    return (wave[mask],a[mask])


if __name__=="__main__":
    with sqlite3.connect("samples.db") as con:
        cur = con.cursor()

        cur.execute('SELECT "image","pixels","bar","name" FROM images INNER JOIN samples ON "index" == "sample";')
        rows = cur.fetchall()
        for i,p,b,n in rows:
            print(i)
            q,a = extract(i,b,p)
            plt.plot(q,a,label=n)
        plt.legend()
        plt.show()
