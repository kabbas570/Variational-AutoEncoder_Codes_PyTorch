from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
           ###########  Dataloader  #############
import SimpleITK as sitk

import matplotlib.pyplot as plt
import numpy as np
NUM_WORKERS=0
PIN_MEMORY=True


DIM_ = 256
def generate_label_4(gt):
        temp_ = np.zeros([4,gt.shape[1],DIM_,DIM_])
        temp_[0:1,:,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:,:][np.where(gt==0)]=1
        return temp_
    
def same_depth(img):
    temp = np.zeros([img.shape[0],17,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp 
   
class Dataset_train(Dataset): 
    def __init__(self, img_folder,gts_folder):
      
        self.img_folder = img_folder
        self.gts_folder = gts_folder
        self.gts = os.listdir(gts_folder)
    
    def __len__(self):
       return len(self.gts)
   
    def __getitem__(self, index):
        img_path = os.path.join(self.img_folder,self.gts[index])
        gt_path = os.path.join(self.gts_folder,self.gts[index])
        
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        img = np.moveaxis(img,(2,1,0),(0,1,2))
        img = np.expand_dims(img, axis=0)
                 
        gt = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt)
        gt = np.moveaxis(gt,(2,1,0),(0,1,2))
        gt = np.expand_dims(gt, axis=0)
        gt = generate_label_4(gt)

        return img,gt
        
def Data_Loader_train(img_folder,gts_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_train(img_folder=img_folder,gts_folder=gts_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

img_folder_val = r'C:\My_Data\M2M Data\data\new_data\SA_Data\train\imgs'
gt_folder_val = r'C:\My_Data\M2M Data\data\new_data\SA_Data\train\gts'

loader = Data_Loader_train(img_folder_val,gt_folder_val,1)

a = iter(loader)

for i in range(160):
    a1 = next(a)

    img = a1[0][0,0,:]
    print(img.shape)
# for i in range(9):
#     plt.figure()
#     plt.imshow(a1[0][0,0,i,:])
    
# for i in range(9):
#     plt.figure()
#     plt.imshow(a1[1][0,2,i,:])
