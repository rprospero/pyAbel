import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fftpack import ifftshift,fftshift
from scipy.cluster.vq import kmeans2

def queryK(image,k):
    mask = np.where(image==k)
    return float(len(mask[0]))/shape[0]/shape[1]

imageFile = "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.13000X.Parnell.0007.jpg"

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

images = ["/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.13000X.Parnell.0002.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.13000X.Parnell.0003.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.13000X.Parnell.0005.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.13000X.Parnell.0007.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.18500X.Parnell.0001.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.18500X.Parnell.0004.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.18500X.Parnell.0006.jpg",
          "/home/me1alw/Science/StructuralColour/MathSlice/Macaw/14-076.80000.0V.18500X.Parnell.0008.jpg"]

data = np.array([vf(i) for i in images])
print(data)
print(np.mean(data))
print(np.std(data))

