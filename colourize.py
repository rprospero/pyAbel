#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift

from abel import imbin, invabel

import sqlite3

import argparse

from colorpy.ciexyz import xyz_from_spectrum
from colorpy.colormodels import rgb_from_xyz, xyz_normalize


def SpectrumToRGB(q, iq):
    x = np.arange(300, 800, 1)
    y = np.interp(x, q[::-1], iq[::-1])
    spec = np.vstack([x, y])
    xyz = xyz_from_spectrum(spec.T)
    return rgb_from_xyz(xyz_normalize(xyz))


def invfourier(image):
    return ifftshift(np.fft.ifftn(image))


def quantize(x):
    middle = np.median(x)
    mask = x >= middle
    x[mask] = 1  # upmean
    x[np.logical_not(mask)] = 0  # downmean
    del mask
    return x


def extract(imageFile, bar, pixels, abel=True):
    image = misc.imread(imageFile, flatten=True)

    transform = invfourier(image)

    size = image.shape[0]
    if size < 200:
        return None

    q, f = imbin(np.abs(transform))
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

    a[a < 0] = 0
    rgb = SpectrumToRGB(wave[mask], a[mask]*scale*1e9)

    r = image * rgb[0]
    g = image * rgb[1]
    b = image * rgb[2]
    temp = np.dstack([r, g, b])
    # print(np.min(image))
    # print(np.max(image))
    # print(np.min(temp))
    # print(np.max(temp))
    # print("max")
    return np.asarray(temp, dtype=np.uint8)


def plotSample(cur, index, abel=True):
    print(index)
    cur.execute('SELECT "image","pixels","bar","name"'
                ' FROM images INNER JOIN samples ON'
                ' "index" == "sample" WHERE "index" == ?;', (index, ))
    rows = cur.fetchall()
    vs = []
    for i, p, b, n in rows:
        values = extract(i, b, p, abel)
        print(i)
        if values is None:
            continue
        vs.append(values)
        plt.imshow(values)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Turn a set of sample images into a spectrum.')
    parser.add_argument('--noAbel', action='store_false',
                        help='Skip the Abel correction on Fourier transform')
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
