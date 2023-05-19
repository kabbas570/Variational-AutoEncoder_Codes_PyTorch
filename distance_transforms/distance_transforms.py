# def SDF_2D(img):
#     img = img.astype(np.uint8)
#     normalized_sdf = np.zeros(img.shape)
#     posmask = img.astype(bool)
#     if posmask.any():
#         negmask = ~posmask
#         posdis = distance(posmask)
#         negdis = distance(negmask)
#         boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
#         sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))*negmask - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))*posmask
#         sdf[boundary==1] = 0
#         normalized_sdf = sdf
#     return normalized_sdf
 

# def SDF_3D(img_gt):
#     img_gt = img_gt.astype(np.uint8)
#     normalized_sdf = np.zeros(img_gt.shape)
#     posmask = img_gt.astype(bool)
#     if posmask.any():
#         negmask = ~posmask
#         posdis = distance(posmask)
#         negdis = distance(negmask)
#         boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
#         sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
#         sdf[boundary==1] = 0
#         normalized_sdf = sdf
#         assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
#         assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

#     return normalized_sdf
