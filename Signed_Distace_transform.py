import imageio
import numpy as np
import matplotlib.pyplot as plt
import skfmm

# Load image
im = imageio.imread(r"C:\Users\Abbas Khan\Desktop\vae_codes\2D_VIZ\INP\1_IMG.png")
ima = np.array(im)
ima =ima[:,:,0]/255

import skfmm

def SDF(ima):
    ima = np.stack((ima,ima,ima),2)
    phi = np.int64(np.any(ima[:, :, :3], axis = 2))
    phi = np.where(phi, 0, -1) + 0.5
    sd = skfmm.distance(phi, dx = 1)
    return sd

a = SDF(ima)

## For 2D 


import skfmm

def SDF_2D(ima):
    phi = np.int64(ima)
    phi = np.where(phi, 0, -1) + 0.5
    sd = skfmm.distance(phi, dx = 1)
    return sd

from scipy import ndimage
dst = ndimage.distance_transform_edt(img)



def get_distance(f):
    """Return the signed distance to the 0.5 levelset of a function."""

    # Prepare the embedding function.
    f = f > 1

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))

    return distance

c =get_distance(img)


for i in range(17):
    plt.figure()
    plt.imshow(c[i,:])
    
    print(np.max(c[i,:]))


from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


def SDF_2D(img):
    img = img.astype(np.uint8)
    normalized_sdf = np.zeros(img.shape)
    posmask = img.astype(bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))*negmask - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))*posmask
        sdf[boundary==1] = 0
        normalized_sdf = sdf
    return normalized_sdf





#import skfmm
#
#def SDF_2D(ima):
#    phi = np.int64(ima)
#    phi = np.where(phi, 0, -1) + 0.5
#    sd = skfmm.distance(phi, dx = 1)
#    return sd

from scipy import ndimage
#def SDF_2D(f):
#    """Return the signed distance to the 0.5 levelset of a function."""
#
#    # Prepare the embedding function.
#    f = f > 0.5
#
#    # Signed distance transform
#    dist_func = ndimage.distance_transform_edt
#    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))
#
#    return distance
