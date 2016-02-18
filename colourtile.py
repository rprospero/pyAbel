#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift

from abel import imbin,fourier,invabel
from imbin import imbin2 as imbin

import sqlite3
import scipy

import argparse

from colorpy.ciexyz import xyz_from_spectrum
from colorpy.colormodels import rgb_from_xyz,xyz_normalize,irgb_from_rgb

def SpectrumToRGB(q,iq):
    #mask = np.logical_and(q > 300e-9,q<800e-9)
    x = np.arange(300,800,1)
    y = np.interp(x,q[::-1],iq[::-1])
    #spec = np.vstack([2*np.pi/q[mask],np.log(iq[mask]*q[mask]**2)])
    #spec = np.vstack([q[mask],iq[mask]])
    spec = np.vstack([x,y])
    xyz = xyz_from_spectrum(spec.T)
    return rgb_from_xyz(xyz_normalize(xyz))
    #return rgb_from_xyz(xyz/4e12)


def invfourier(image):
    return ifftshift(np.fft.ifftn(image))

def quantize(x):
    middle = np.median(x)
    mask = x >= middle
    upmean = np.mean(x[mask])
    downmean = np.mean(x[np.logical_not(mask)])
    x[mask] = 1 #upmean
    x[np.logical_not(mask)] = 0 # downmean
    del mask
    return x

def extract(image,bar,pixels,abel=True):

    #image = quantize(image)

    transform = invfourier(image)

    pi = np.pi
    size = image.shape[0]
    if size < 25:
        return None

    q,f = imbin(np.abs(transform))
    if abel:
        a = invabel(f)
    else:
        a = f

    scale = bar/pixels

    index = 1.54
    fraction = 0.0
    index = 1 + fraction * (index-1)
    
    wave = 2*scale*size/(q+1)*index
    mask = np.logical_and(wave > 40, wave < 1000)
    
    #wave = wave[::-1]
    #a = a[::-1]

    a[a<0] = 0
    #plt.plot(wave[mask],a[mask]*scale*10)
    #plt.show()
    rgb = SpectrumToRGB(wave[mask],a[mask]*scale*1e9)
    # print(rgb)
    #temp = np.reshape(image,(image.shape[0],image.shape[1],1))*rgb#*np.reshape(np.array([1,1,1]),(1,1,3))
    r = image * rgb[0]
    g = image * rgb[1]
    b = image * rgb[2]
    temp = np.dstack([r,g,b])
    # print(np.min(image))
    # print(np.max(image))
    # print(np.min(temp))
    # print(np.max(temp))
    # print("max")
    return np.asarray(temp,dtype=np.uint8)

def plotSample(cur,index,abel=True):
    # print(index)
    subs = 9
    cur.execute('SELECT "image","pixels","bar","name" FROM images INNER JOIN samples ON "index" == "sample" WHERE "index" == ?;',(index,))
    rows = cur.fetchall()
    hasPlot = False
    accum = []
    xs = np.arange(100,1000)
    vs = []
    for i,p,b,n in rows:
        image = misc.imread(i,flatten=True)
        shape = image.shape
        resize = shape[0]-shape[0]%subs
        image = image[0:resize,
                      0:resize]
        # print(shape,shape[0],shape[0]%subs,resize)
        hs = []
        for h in np.hsplit(image,subs):
            # print(len(h))
            tiles = []
            for tile in np.vsplit(h,subs):
                # print("tile",tile.shape)
                values = extract(tile,b,p,abel)
                if values is None:
                    print("No Values!")
                    continue
                tiles.append(values)
            hs.append(np.vstack(tiles))
        result = np.hstack(hs)
        vs.append(result)
        plt.imshow(result, extent=[0, result.shape[0]*b/p/1000,
                                   0, result.shape[1]*b/p/1000])
        print(i)
        plt.show()
    #vs = np.reshape(vs,(-1,1,3))
    #plt.imshow(vs)
    #plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Turn a set of sample images into a spectrum.')
    parser.add_argument('--noAbel', action='store_false',
                        help='Skip the Abel correction on the Fourier transform')
    parser.add_argument('sample', action='store',
                        help='The index for the sample being examined')
    parser.add_argument('file', action='store',
                        help='Where to save the spectrum')

    args = parser.parse_args()
    print(args)

    with sqlite3.connect("samples.db") as con:
        cur = con.cursor()

        cur.execute('SELECT "index" FROM samples;')
        indices = cur.fetchall()

        indices = [(args.sample,)]

        values = [plotSample(cur,index[0],args.noAbel)
                 for index in indices]
        values = [v for v in values if v is not None]

        xs = np.arange(100,1000)
        for title,value in values:
            np.savetxt(args.file,np.vstack([xs,value]).T)
        #plt.legend()
        #plt.show()
