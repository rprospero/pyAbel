#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift
from scipy.cluster.vq import kmeans2
import sqlite3
from sys import argv

def queryK(image,k):
    shape = image.shape
    mask = np.where(image==k)
    return float(len(mask[0]))/shape[0]/shape[1]

def vf(imageFile):

    image = misc.imread(imageFile)
    image = np.mean(image,axis=2)
    shape = image.shape

    image = np.reshape(image,shape[0]*shape[1])
    values,means = kmeans2(image,2)
    means = np.reshape(means,shape)
    image = np.reshape(image,shape)

    ks = []
    for k,v in enumerate(values):
        ks.append((v,queryK(means,k)))
        ks.sort(key=lambda x: x[0])

    return ks[-1][1]

if __name__ == "__main__":
    index = int(argv[1])

    with sqlite3.connect("samples.db") as con:
        cur = con.cursor()
        cur.execute('SELECT "image" FROM images INNER JOIN samples ON "index" == "sample" WHERE "index" == ?;',(index,))

        images = cur.fetchall()
        data = np.array([vf(i) for (i,) in images])
    
        print(str(np.mean(data)) + "±" + str(np.std(data)))
