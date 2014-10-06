import numpy as np

def imbin(image):
    """Take an n-dimensional array and return a 1D histogram
    of the radial integration"""
    dims = np.array(image.shape)
    dims /= 2

    values = np.zeros(sum(dims))
    pixels = np.zeros(sum(dims))

    indices = np.indices(tuple(image.shape))
    rs = np.zeros(image.shape, dtype=np.float32)
    for (index, dim) in zip(indices, dims):
        rs += (index-dim)**2
    rs = np.asarray(np.round(np.sqrt(rs)), dtype=np.int32)

    bins = np.arange(0, np.max(rs)+1)
    values, _ = np.histogram(rs, bins=bins, weights=image)
    pixels, _ = np.histogram(rs, bins=bins)

    values /= pixels

    x = np.arange(0, np.max(rs))

    return (x, values[:len(x)])

def imbin2(image,scale=None):
    """Take an n-dimensional array and return a 1D histogram
    of the radial integration"""
    dims = np.array(image.shape)
    dims /= 2

    if scale is None:
        scale = [1 for x in image.shape]

    values = np.zeros(sum(dims))
    pixels = np.zeros(sum(dims))

    indices = np.indices(tuple(image.shape))
    rs = np.zeros(image.shape, dtype=np.float32)
    for (index,dim,scl) in zip(indices,dims,scale):
        rs += (scl*(index-dim))**2
    rs = np.asarray(np.round(np.sqrt(rs)), dtype=np.int32)

    bins = np.arange(0, np.max(rs)+1)
    values, _ = np.histogram(rs, bins=bins, weights=image)
    pixels, _ = np.histogram(rs, bins=bins)

    values /= pixels

    x = np.arange(0, np.max(rs))

    return (x, values[:len(x)])
