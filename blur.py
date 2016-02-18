#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abel import imbin, invabel
from scipy.fftpack import fftshift
from scipy.ndimage.filters import gaussian_filter

from extract import quantize, invfourier


def fourier(image):
    return np.fft.fftn(fftshift(image))

LOW = 23
HIGH = 26

def plot_image(image):
    plt.subplot(121)
    left = plt.imshow(image, cmap=plt.cm.gray)

    plt.subplot(122)
    plt.ylim(0,2)
    qs, Is = imbin(np.abs(invfourier(image)))
    qs = qs[2:]
    Is = Is[2:]
    mask = qs < 5*HIGH
    Is /= np.max(Is[LOW-5:HIGH+5])
    As = invabel(Is)
    As /= np.max(As[LOW-5:HIGH+5])

    qs = qs[mask]
    Is = Is[mask]
    As = As[mask]

    right = plt.plot(qs, Is, "k-",
                     qs, As, "c-")
    return (left, right)


def main():
    SIZE = 301
    DIM = (SIZE, SIZE, SIZE)
    base = np.zeros(DIM, dtype=np.complex64)
    Is = np.indices(DIM, dtype=np.float64)
    rs = (Is[0] - (SIZE - 1) / 2)**2
    for ix in Is[1:]:
        rs += (ix-(SIZE-1)/2)**2
    rs = np.sqrt(rs)

    mask = np.logical_and(rs >= LOW,
                          rs < HIGH)
    base[mask] = 1

    phase = np.exp(2j*np.pi*np.random.random(DIM))
    base *= phase

    real = fourier(base)

    print(np.real(real).dtype)

    # image = quantize(np.real(real[:, :, 10]))
    image = np.real(real[:, :, 10])

    # plt.imshow(quantize(image), cmap=plt.cm.spectral)

    # plt.imshow(np.real(invfourier(image)))

    fig = plt.figure()
    ims = []
    for i in np.arange(0, 6, 0.01):
        l, r = plot_image(quantize(gaussian_filter(image, i)))
        # l, r = plot_image(image+i*2*(np.random.random((SIZE, SIZE))-0.5))
        # ims.append(l)
        r.append(l)
        ims.append(r)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=9600)

    im_ani = animation.ArtistAnimation(fig, ims, interval=25,
                                       repeat_delay=1000,
                                       blit=True)
    plt.show()
    # im_ani.save('noise.mp4', writer=writer)

    del im_ani


main()
