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
