#!/usr/bin/python

import numpy as np
from scipy import misc
from scipy.fftpack import ifftshift

from abel import imbin, invabel

import sqlite3
import scipy

import argparse


def invfourier(image):
    return ifftshift(np.fft.ifftn(image))


def quantize(x):
    middle = np.median(x)
    mask = x >= middle
    upmean = np.mean(x[mask])
    downmean = np.mean(x[np.logical_not(mask)])
    x[mask] = 1  # upmean
    x[np.logical_not(mask)] = 0  # downmean
    print(upmean)
    print(downmean)
    del mask
    return x


def extract(imageFile, bar, pixels, abel=True):
    image = misc.imread(imageFile, flatten=True)
    print(image.shape)

    transform = invfourier(image)

    size = image.shape[0]
    if size < 200:
        return None

    q, f = imbin(np.abs(transform))
    if abel:
        a = invabel(f)
    else:
        a = f

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

    return (wave[mask], a[mask]*scale)


def plotSample(cur, index, abel=True):
    cur.execute('SELECT "image","pixels","bar","name"'
                'FROM images INNER JOIN samples ON "index" '
                '== "sample" WHERE "index" == ?;', (index, ))
    rows = cur.fetchall()
    hasPlot = False
    accum = []
    xs = np.arange(100, 1000)
    for i, p, b, n in rows:
        values = extract(i, b, p, abel)
        if values is None:
            continue
        q, a = values
        hasPlot = True
        print(i)
        accum.append(scipy.interp(xs, q[::-1], a[::-1]))
    if hasPlot:
        average = np.median(np.vstack(accum), axis=0)
        return (n, average)
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Turn a set of sample images into a spectrum.')
    parser.add_argument('--noAbel', action='store_false',
                        help='Skip the Abel correction'
                             ' on the Fourier transform')
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

        values = [plotSample(cur, index[0], args.noAbel)
                  for index in indices]
        values = [v for v in values if v is not None]

        xs = np.arange(100, 1000)
        for title, value in values:
            np.savetxt(args.file, np.vstack([xs, value]).T)
