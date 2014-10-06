#!/usr/bin/python

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift
from scipy import misc
from imbin import imbin

def invabel(x):
    result = np.zeros(len(x),dtype = np.float64)
    #Let's just focus on making it work for now
    for r in range(len(x)):
        for y in np.arange(r+1,len(x)-1):
            dy = x[y+1]-x[y]
            result[r] += dy/np.sqrt((y-0.5)**2-r**2)
    return -result/np.pi

def fourier(image):
    return fftshift(np.abs(np.fft.fft2(image)))


if __name__=="__main__":
    infile = sys.argv[1]

    image = misc.imread(infile,flatten=True)
    image /= 255

    transform = fourier(image)
    radius,radial = imbin(transform)
    adjusted = invabel(radius[5:])

    plt.plot(radius,radial,label="Fourier")
    plt.plot(radius[5:],adjusted,label="Abel")

    plt.legend()
    plt.show()
    #plt.savefig("~/test.pdf")
    #plt.clf()
