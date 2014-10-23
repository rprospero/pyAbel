#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

f = np.loadtxt("macaw_fourier.txt")
a = np.loadtxt("macaw_abel.txt")

fx = f[:,0]
ax = a[:,0]

index = 1.517
indexb = 8800.0
fraction = 0.3
#fraction = 0.3

fx *= 1 + fraction * (index-1+indexb/fx/fx)
ax *= 1 + fraction * (index-1+indexb/fx/fx)

fmask = np.logical_and(fx>=350,fx<=800)
amask = np.logical_and(ax>=350,ax<=800)

f = f[:,1]
a = a[:,1]

f /= np.max(f[fmask])
a /= np.max(a[amask])

plt.plot(fx,f,color="grey",label="Without Abel",linewidth=4)
plt.plot(ax,a,color="black",label="With Abel",linewidth=4)
plt.ylabel("Normalized Spectrum",fontsize=24)
plt.xlabel("Wavelength",fontsize=24)

refl = np.loadtxt("macaw_reflectance.txt",skiprows=1)

plt.plot(refl[:,0],refl[:,1]/np.max(refl[:,1]),
         color="cyan",linewidth=4,
         label="Reflectance")

plt.title("Macaw Feather",fontsize=36)
#plt.axvline((450+495)/2,color="blue",linewidth=4)
#plt.axvline((495+570)/2,color="green",linewidth=4)
#plt.axvline((620+740)/2,color="red",linewidth=4)
plt.ylim(0,1)
plt.xlim(350,800)
plt.legend()
plt.show()
