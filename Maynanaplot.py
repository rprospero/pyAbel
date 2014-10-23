#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

f = np.loadtxt("Fourier.txt")
a = np.loadtxt("Abeled.txt")

fx = f[:,0]
ax = a[:,0]

index = 1.54
fraction = 0.5

fx *= 1 + fraction * (index-1)
ax *= 1 + fraction * (index-1)

f = f[:,1]
a = a[:,1]

f /= np.max(f)
a /= np.max(a)

plt.plot(fx,f,color="grey",label="Without Abel",linewidth=4)
plt.plot(ax,a,color="black",label="With Abel",linewidth=4)
plt.ylabel("Normalized Spectrum",fontsize=24)
plt.xlabel("Wavelength",fontsize=24)

plt.title("Possible Maynana Feather",fontsize=36)
plt.axvline((450+495)/2,color="blue",linewidth=4)
plt.axvline((495+570)/2,color="green",linewidth=4)
plt.axvline((620+740)/2,color="red",linewidth=4)
plt.ylim(0,1)
plt.legend()
plt.show()
