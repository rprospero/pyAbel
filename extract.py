#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift

from abel import imbin,fourier,invabel
from imbin import imbin2 as imbin
from structure3d import animate_plot

def invfourier(image):
    return np.fft.ifftn(ifftshift(image))

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

def rotate_plane(size,theta,phi,psi):
    xs = np.fromiter((x for x in range(size) for y in range(size)),int)
    ys = np.fromiter((y for x in range(size) for y in range(size)),int)
    zs = np.fromiter((0 for x in range(size) for y in range(size)),int)
    xs -= int(size/2)
    ys -= int(size/2)
    #zs -= int(size/2)
    grid = np.vstack((xs,ys,zs))
    #print(grid)
    #print("---")
    Rx = np.array([[1,0,0],
                   [0,np.cos(theta),-np.sin(theta)],
                   [0,np.sin(theta),np.cos(theta)]])
    Rz = np.array([[np.cos(phi),-np.sin(phi),0],
                   [np.sin(phi),np.cos(phi),0],
                   [0,0,1]])
    Rx2 = np.array([[1,0,0],
                    [0,np.cos(psi),-np.sin(psi)],
                    [0,np.sin(psi),np.cos(psi)]]) 
    R = np.dot(np.dot(Rx,Rz),Rx2)
    grid = np.asarray(np.round(np.dot(R,grid)),dtype=np.int32)
    #print(grid)
    #print("---")
    #print(grid.shape)
    grid[0,:] += int(size/2)
    grid[1,:] += int(size/2)
    return grid

def slice_plane(structure,theta,phi,psi,d):
    size = structure.shape[0]
    image = np.zeros((size,size))
    temp = rotate_plane(size,theta,phi,psi)
    temp += np.array([d]).T
    temp[temp >= size] -= size
    temp[temp < 0] += size
    for u in range(size):
        for v in range(size):
            image[u,v] = structure[tuple(temp.T[size*u+v])]
    return image

if __name__=="__main__":
    size = 201
    wavelength = 40
    tol = 2.5

    spec = np.zeros((size,size,size),
                    dtype=np.complex128)

    factors = [16,1,4]
    rs = sum([(index - (size-1)/2)**2/factor
              for index,factor in zip(np.indices((size, size, size)),
                               factors)])

    rs = np.round(np.sqrt(rs))

    mask = np.logical_and(rs >= (1-tol/100.0)*wavelength,
                          rs <= (1+tol/100.0)*wavelength)
    del rs
    spec[mask] = 1

    spec *= np.exp(2*np.pi*1.0j*np.random.random((size,size,size)))

    #animate_plot(np.abs(spec))

    q,sq = imbin(np.abs(spec))
    
    ### Test Section
    # import matplotlib.animation as animation

    # fig2 = plt.figure()
    # vmin = np.min(spec)
    # vmax = np.max(spec)
    # ims = []
    # for i in np.arange(0,2*np.pi,0.1):
    #     theta = 0
    #     phi = np.pi/2
    #     psi = i
    #     image = slice_plane(spec,theta,
    #                         phi,psi,[0,0,int(size/2)])
    #     ims.append([plt.imshow(image,vmin=vmin,vmax=vmax)])
    #     print(i)
    # im_ani = animation.ArtistAnimation(fig2, ims, interval=50,
    #                                    repeat_delay=0,
    #                                    blit=True)

    # plt.show()

    # del im_ani
    

    real = invfourier(spec)
    real = quantize(np.real(real))

    animate_plot(real/np.max(real[3:]))

    #plt.plot(q,sq/np.max(sq),label="Original")
    _,sq = imbin(np.abs(spec))
    plt.plot(q,sq/np.max(sq),label="Original")

    #Z Slices
    
    slices = []
    trials = []
    for i in range(size):
        slce = fftshift(np.abs(np.fft.fft2(real[:,:,i])))
        q,f = imbin(slce)
        slices.append(f)

    fs = np.mean(np.vstack(slices),axis=0)
    fs /= np.max(fs[1:])

    abels = invabel(fs)
    abels /= np.max(np.abs(abels[3:]))

    plt.plot(q+1,abels,label="Abel Z")

    #Y Slices
    
    slices = []
    trials = []
    for i in range(size):
        slce = fftshift(np.abs(np.fft.fft2(real[:,i,:])))
        q,f = imbin(slce)
        slices.append(f)

    fs = np.mean(np.vstack(slices),axis=0)
    fs /= np.max(fs[1:])

    abels = invabel(fs)
    abels /= np.max(np.abs(abels[3:]))

    plt.plot(q+1,abels,label="Abel Y")
    
    #X Slices
    
    slices = []
    trials = []
    for i in range(size):
        slce = fftshift(np.abs(np.fft.fft2(real[i,:,:])))
        q,f = imbin(slce)
        slices.append(f)

    fs = np.mean(np.vstack(slices),axis=0)
    fs /= np.max(fs[1:])

    abels = invabel(fs)
    abels /= np.max(np.abs(abels[3:]))

    plt.plot(q+1,abels,label="Abel X")

    #X Mixed
    
    slices = []
    trials = []
    for i in range(size):
        slce = fftshift(np.abs(np.fft.fft2(real[i,:,:])))
        q,f = imbin(slce)
        slices.append(f)
        slce = fftshift(np.abs(np.fft.fft2(real[:,i,:])))
        q,f = imbin(slce)
        slices.append(f)
        slce = fftshift(np.abs(np.fft.fft2(real[:,:,i])))
        q,f = imbin(slce)
        slices.append(f)

    fs = np.mean(np.vstack(slices),axis=0)
    fs /= np.max(fs[1:])

    abels = invabel(fs)
    abels /= np.max(np.abs(abels[1:]))

    plt.plot(q+1,abels,label="Abel Mixed Orientations")

    slices = []
    for i in range(500):
        theta = np.random.random((1))[0] * 2 * np.pi
        phi = np.random.random((1))[0] * 2 * np.pi
        psi = np.random.random((1))[0] * 2 * np.pi

        image = slice_plane(real,theta,
                            phi,psi,[0,0,int(size/2)]
                            +wavelength*(np.random.random(3)-0.5))
        slce = fftshift(np.abs(np.fft.fft2(image)))
        q,f = imbin(slce)
        #f = invabel(f)
        slices.append(f)
        

    fs = np.mean(np.vstack(slices),axis=0)
    fs /= np.max(np.abs(fs))

    abels = invabel(fs)
    abels /= np.max(np.abs(abels[3:]))

    plt.plot(q+1,abels,label="Abel Random Orientation")


    plt.ylim(-0.5,1)
    plt.legend()
    plt.show()
