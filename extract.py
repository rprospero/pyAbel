#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift

from abel import imbin,fourier,invabel
from imbin import imbin2 as imbin

import sqlite3
import scipy

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

    image = quantize(image)

    transform = invfourier(image)

    pi = np.pi
    size = image.shape[0]
    if size < 200:
        return None

    q,f = imbin(np.abs(transform))
    a = invabel(f)

    print(bar)
    print(pixels)
    scale = bar/pixels

    print(scale)
    print(size)
    print(scale*size)
    index = 1.54
    fraction = 0.0
    index = 1 + fraction * (index-1)
    
    wave = 2*scale*size/(q+1)*index
    mask = np.logical_and(wave > 40, wave < 1000)

    return (wave[mask],a[mask]*scale)

def plotSample(cur,index):
    cur.execute('SELECT "image","pixels","bar","name" FROM images INNER JOIN samples ON "index" == "sample" WHERE "index" == ?;',(index,))
    rows = cur.fetchall()
    hasPlot = False
    accum = []
    xs = np.arange(350,750)
    for i,p,b,n in rows:
        values = extract(i,b,p)
        if values is None:
            continue
        q,a = values
        hasPlot = True
        print(i)
        accum.append(scipy.interp(xs,q[::-1],a[::-1]))
        #plt.plot(q,a,label=i)
    if hasPlot:
        average = np.median(np.vstack(accum),axis=0)
        #plt.plot(xs,average,label="Average")
        #plt.title(n)
        #plt.legend()
        #plt.show()
        return (n,average)
    else:
        return None


if __name__=="__main__":
    with sqlite3.connect("samples.db") as con:
        cur = con.cursor()

        cur.execute('SELECT "index" FROM samples;')
        indices = cur.fetchall()

        indices = [(7,)]

        values = [plotSample(cur,index[0])
                 for index in indices]
        values = [v for v in values if v is not None]

        xs = np.arange(350,750)
        for title,value in values:
            plt.plot(xs,value,label=title)
            np.savetxt("Abeled.txt",np.vstack([xs,value]).T)
        plt.legend()
        plt.show()
