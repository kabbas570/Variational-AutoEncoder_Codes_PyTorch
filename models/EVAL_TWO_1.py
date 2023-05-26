import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
            
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            # nn.InstanceNorm2d(mid_channels,affine=True),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.01 , inplace=True),
            #nn.GELU(),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.01 , inplace=True),
            #nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class unet_2d(nn.Module):
    def __init__(self, n_channels = 4, bilinear=False):
        super(unet_2d, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor)
        
        self.up0 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32,4)
                                
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        z1 = self.up0(x6, x5)
        z2 = self.up1(z1, x4)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)
        logits1 = self.outc(z5)
         
        return logits1
    
import torch
import torch.nn as nn
import torchvision
class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels,stride,kernel_size):
        super().__init__()
        
        self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU()
        )

    def forward(self, x1):
        x = self.up(x1)
        return  x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)
class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x)
            
Base = 64
class Autoencoder_2D2_1(nn.Module):
    def __init__(self, n_channels = 3, bilinear=False):
        super(Autoencoder_2D2_1, self).__init__()
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            
        Conv(n_channels, Base,1,1,3),
        #Conv(Base, Base,1,1,3),
        Conv(Base, Base,2,1,3),
        
        Conv(Base, 2*Base,1,1,3),
        #Conv(2*Base, 2*Base,1,1,3),
        Conv(2*Base, 2*Base,2,1,3),
        
        Conv(2*Base,4*Base,1,1,3), 
        #Conv(4*Base,4*Base,1,1,3), 
        Conv(4*Base, 4*Base,2,1,3), 
        
        Conv(4*Base,8*Base,1,1,3), 
        #Conv(8*Base,8*Base,1,1,3), 
        Conv(8*Base, 8*Base,2,1,3), 

        )
        
        self.decoder =  nn.Sequential(
            
        Deconv(8*Base,8*Base,2,2),
        #Conv(8*Base,8*Base,1,1,3),
        Conv(8*Base,8*Base,1,1,3),
        
        Deconv(8*Base,4*Base,2,2),
        #Conv(4*Base,4*Base,1,1,3),
        Conv(4*Base,4*Base,1,1,3),
        
        Deconv(4*Base,2*Base,2,2),
        #Conv(2*Base,2*Base,1,1,3),
        Conv(2*Base,2*Base,1,1,3),
        
        Deconv(2*Base,Base,2,2),
        #Conv(Base,Base,1,1,3),
        Conv(Base,Base,1,1,3),
        
        OutConv_2d(Base,3),
        )
        
        #self.activation = torch.nn.Sigmoid()

    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      #return self.activation(encoded)
      return encoded

# Input_Image_Channels = 3
# def model() -> Autoencoder_2D2_1:
#     model = Autoencoder_2D2_1()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#from scipy.ndimage.morphology import distance_transform_edt as edt
import nibabel as nib
import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import SimpleITK as sitk
import cv2
from typing import List, Union, Tuple

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256

   
def same_depth(img):
    temp = np.zeros([img.shape[0],20,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
   
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_
   
# def generate_label_4(gt):
#         temp_ = np.zeros([4,gt.shape[1],DIM_,DIM_])
#         temp_[0:1,:,:,:][np.where(gt==1)]=1
#         temp_[1:2,:,:,:][np.where(gt==2)]=1
#         temp_[2:3,:,:,:][np.where(gt==3)]=1
#         temp_[3:4,:,:,:][np.where(gt==0)]=1
#         return temp_

def generate_label_3_gt(gt):
        temp_ = np.zeros([3,gt.shape[1],DIM_,DIM_])
        temp_[0:1,:,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:,:][np.where(gt==3)]=1
        return temp_
    
    
kernel = np.ones((2,2),np.uint8)
def generate_label_3_proj(gt):
        temp_ = np.zeros([3,DIM_,DIM_])
        temp_[0,:][np.where((gt>0.5) & (gt <1.2))]=1
        temp_[1,:][np.where((gt>1.2) & (gt <2.2))]=1
        temp_[2,:][np.where((gt>2.2) & (gt <3.2))]=1
        temp_[2,:] = cv2.morphologyEx(temp_[2,:], cv2.MORPH_OPEN, kernel)
        temp_[1,:] = cv2.morphologyEx(temp_[1,:], cv2.MORPH_OPEN, kernel)
        temp_[0,:] = cv2.morphologyEx(temp_[0,:], cv2.MORPH_OPEN, kernel)
        return temp_
    

def Normalization_LA_ES(img):
        img = (img-114.8071)/191.2891
        return img 
def Normalization_LA_ED(img):
        img = (img-114.7321)/189.8573
        return img 
        
def Normalization_SA_ES(img):
        img = (img-62.5983)/147.4826
        return img 
def Normalization_SA_ED(img):
        img = (img-62.9529)/147.6579
        return img 

def resample_image(image: sitk.Image ,
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
                
        out_spacing = (1.25, 1.25,original_spacing[2])

        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            #out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
            out_size = (np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)

        else:
            out_size = np.array(out_size)
            
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        
        return resample.Execute(image)
    
def SA_to_LA(SA_img,LA_img):    ## new LA image 
    size = (LA_img.GetSize()) 
    new_img = sitk.Image(LA_img)
    
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            for z in range(0,size[2]):
                new_img[x,y,z] = 0
                point = LA_img.TransformIndexToPhysicalPoint([x,y,z])  ##  determine the physical location of a pixel:
                index_LA = SA_img.TransformPhysicalPointToIndex(point)
                if index_LA[0] < 0 or index_LA[0]>= SA_img.GetSize()[0]:
                    continue
                if index_LA[1] < 0 or index_LA[1]>= SA_img.GetSize()[1]:
                    continue
                if index_LA[2] < 0 or index_LA[2]>= SA_img.GetSize()[2]:
                    continue
                new_img[x,y,z] = SA_img[index_LA[0],index_LA[1],index_LA[2]]
    
    return new_img

def generate_label_4(gt):
        temp_ = np.zeros([4,gt.shape[1],DIM_,DIM_])
        temp_[0:1,:,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:,:][np.where(gt==0)]=1
        return temp_
    
class Dataset_V(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
       
        #### LA PART ### LA PART ###
        
        img_SA_path = img_path+'_LA_ES.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        img = Normalization_LA_ES(img)
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        img_LA_ES = np.expand_dims(img, axis=0)
        
        
        ## LA_ES_gt ####
        img_SA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        img = np.expand_dims(img, axis=0)
        temp_LA_ES = generate_label_4(img) 
        
        ## LA_ED_img ####
        img_SA_path = img_path+'_LA_ED.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2]
        img = Normalization_LA_ED(img)
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        img_LA_ED = np.expand_dims(img, axis=0)
        
        ## LA_ED_gt ####
        img_SA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2]
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        img = np.expand_dims(img, axis=0)
        temp_LA_ED = generate_label_4(img) 
        
        
        # ### sa_to_la_ES mapping ####
        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_SA_path = img_path_SA +'_SA_ES_gt.nii.gz'
        img_SA_1 = sitk.ReadImage(img_SA_path)
        
        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path = img_path_LA +'_LA_ES_gt.nii.gz'
        img_LA_1 = sitk.ReadImage(img_LA_path)
        
        new_SA_img = SA_to_LA(img_SA_1,img_LA_1)
        new_SA_img = resample_image(new_SA_img)
        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
        
        org_dim3 = new_SA_img.shape[0]
        org_dim1 = new_SA_img.shape[1]
        org_dim2 = new_SA_img.shape[2] 
        proj_2d_ES1 = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
        proj_2d_ES = generate_label_3_proj(proj_2d_ES1[0,:])
        
        # ### sa_to_la_ED mapping ####
        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_SA_path = img_path_SA +'_SA_ED_gt.nii.gz'
        img_SA_1 = sitk.ReadImage(img_SA_path)
        
        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path = img_path_LA +'_LA_ED_gt.nii.gz'
        img_LA_1 = sitk.ReadImage(img_LA_path)
        
        new_SA_img = SA_to_LA(img_SA_1,img_LA_1)
        new_SA_img = resample_image(new_SA_img)
        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
        
        org_dim3 = new_SA_img.shape[0]
        org_dim1 = new_SA_img.shape[1]
        org_dim2 = new_SA_img.shape[2] 
        proj_2d_ED1 = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
        proj_2d_ED = generate_label_3_proj(proj_2d_ED1[0,:])


        return img_LA_ES[:,0,:],img_LA_ED[:,0,:],temp_LA_ES[:,0,:],temp_LA_ED[:,0,:],proj_2d_ES,proj_2d_ED
        
def Data_Loader_V(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader



val_imgs=r'C:\My_Data\M2M Data\data\data_2\single\imgs'
val_csv_path= r"C:\My_Data\M2M Data\data\data_2\single\val1.csv"
df_val = pd.read_csv(val_csv_path)
val_loader = Data_Loader_V(df_val,val_imgs,batch_size = 1)


target_dir1  = '/data/scratch/acw676/Seg_A/vae_viz/'
Threshld = 0.5      
def check_Dice_Score1(loader, model1, device=DEVICE):
    
    Dice_score_LA_RV_ES = 0
    Dice_score_LA_MYO_ES = 0
    Dice_score_LA_LV_ES = 0
    
    Dice_score_LA_RV_ED = 0
    Dice_score_LA_MYO_ED = 0
    Dice_score_LA_LV_ED = 0

    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx,  (temp_LA_ES,temp_LA_ED,proj_2d_ES,proj_2d_ED) in enumerate(loop):
        
        
        temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
        temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
        proj_2d_ES = proj_2d_ES.to(device=DEVICE,dtype=torch.float)
        proj_2d_ED = proj_2d_ED.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 
            
            out_2D_ES = model1(proj_2d_ES)
            out_2D_ES = torch.sigmoid(-1500*out_2D_ES)
            out_2D_ES = (out_2D_ES > 0.5)*1
            out_LA_ES = out_2D_ES
            out_LA_ES_LV = out_LA_ES[:,0:1,:]
            out_LA_ES_MYO = out_LA_ES[:,1:2,:]
            out_LA_ES_RV = out_LA_ES[:,2:3,:]
            
            
            out_2D_ED = model1(proj_2d_ED)
            out_2D_ED = torch.sigmoid(-1500*out_2D_ED)
            out_2D_ED = (out_2D_ED > 0.5)*1
            out_LA_ED = out_2D_ED 
            out_LA_ED_LV = out_LA_ED[:,0:1,:]
            out_LA_ED_MYO = out_LA_ED[:,1:2,:]
            out_LA_ED_RV = out_LA_ED[:,2:3,:]
            
             
            ## Dice Score for ES-LA ###

            single_LA_LV_ES = (2 * (out_LA_ES_LV * temp_LA_ES[:,0:1,:]).sum()) / (
               (out_LA_ES_LV + temp_LA_ES[:,0:1,:]).sum() + 1e-8)
           
            Dice_score_LA_LV_ES +=single_LA_LV_ES
           
            single_LA_MYO_ES = (2 * (out_LA_ES_MYO*temp_LA_ES[:,1:2,:]).sum()) / (
   (out_LA_ES_MYO + temp_LA_ES[:,1:2,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ES += single_LA_MYO_ES

            single_LA_RV_ES = (2 * (out_LA_ES_RV* temp_LA_ES[:,2:3,:]).sum()) / (
       (out_LA_ES_RV + temp_LA_ES[:,2:3,:]).sum() + 1e-8)
            Dice_score_LA_RV_ES += single_LA_RV_ES
            
            ## Dice Score for ED-LA ###

            single_LA_LV_ED = (2 * (out_LA_ED_LV * temp_LA_ED[:,0:1,:]).sum()) / (
                (out_LA_ED_LV + temp_LA_ED[:,0:1,:]).sum() + 1e-8)
            
            Dice_score_LA_LV_ED +=single_LA_LV_ED
            
            single_LA_MYO_ED = (2 * (out_LA_ED_MYO*temp_LA_ED[:,1:2,:]).sum()) / (
    (out_LA_ED_MYO + temp_LA_ED[:,1:2,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ED += single_LA_MYO_ED

            single_LA_RV_ED = (2 * (out_LA_ED_RV* temp_LA_ED[:,2:3,:]).sum()) / (
        (out_LA_ED_RV + temp_LA_ED[:,2:3,:]).sum() + 1e-8)
            Dice_score_LA_RV_ED += single_LA_RV_ED
                  
    Dice_RV_LA = (Dice_score_LA_RV_ES + Dice_score_LA_RV_ED)/2
    Dice_MYO_LA = (Dice_score_LA_MYO_ES + Dice_score_LA_MYO_ED)/2
    Dice_LV_LA = (Dice_score_LA_LV_ES + Dice_score_LA_LV_ED)/2
    
    print(f"Dice_RV_LA  : {Dice_RV_LA/len(loader)}")
    print(f"Dice_MYO_LA  : {Dice_MYO_LA/len(loader)}")
    print(f"Dice_LV_LA  : {Dice_LV_LA/len(loader)}")

    #Overall_Dicescore_LA =( Dice_RV_LA + Dice_MYO_LA + Dice_LV_LA ) /3
    
    out_LA_ES_LV = out_LA_ES[:,0:1,:].numpy()
    out_LA_ES_MYO = out_LA_ES[:,1:2,:].numpy()
    out_LA_ES_RV = out_LA_ES[:,2:3,:].numpy()
    


    return out_LA_ES_LV,out_LA_ES_MYO,out_LA_ES_RV,temp_LA_ES[:,0:1,:],temp_LA_ES[:,1:2,:],temp_LA_ES[:,2:3,:]
    


def gen_one_hot(LV,MYO,RV):
    
    temp =np.zeros([256,256])
    
    temp[np.where(LV==1)]=1
    temp[np.where(MYO==1)]=2
    temp[np.where(RV==1)]=3
    
    return temp
    


Threshld = 0.5      
def check_Dice_Score(loader, model1,model2, device=DEVICE):
    
    Dice_score_LA_RV_ES = 0
    Dice_score_LA_MYO_ES = 0
    Dice_score_LA_LV_ES = 0
    
    Dice_score_LA_RV_ED = 0
    Dice_score_LA_MYO_ED = 0
    Dice_score_LA_LV_ED = 0

    loop = tqdm(loader)
    model1.eval()
    model2.eval()
    
    for batch_idx,  (img_LA_ES,img_LA_ED,temp_LA_ES,temp_LA_ED,proj_2d_ES,proj_2d_ED) in enumerate(loop):
        
        temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
        temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
        proj_2d_ES = proj_2d_ES.to(device=DEVICE,dtype=torch.float)
        proj_2d_ED = proj_2d_ED.to(device=DEVICE,dtype=torch.float)
        img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)
        img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 
            
            out_2D_ES = model1(proj_2d_ES)
            out_2D_ES = torch.sigmoid(-1500*out_2D_ES)
            out_2D_ES = (out_2D_ES >= 0.5)*1

        
            conc_img_ES = torch.cat([img_LA_ES, out_2D_ES], dim=1)
            
            #conc_img_ES = torch.cat([img_LA_ES, temp_LA_ES[:,0:3,:]], dim=1)
            
            
            
            out_LA_ES = model2(conc_img_ES)
            
            
            out_LA_ES = (out_LA_ES > 0.5)*1
            
            #out_LA_ED = .sigmoid(out_LA_ED)
            
            
            out_LA_ES_LV = out_LA_ES[:,0:1,:]
            out_LA_ES_MYO = out_LA_ES[:,1:2,:]
            out_LA_ES_RV = out_LA_ES[:,2:3,:]
            

            gt =  gen_one_hot(temp_LA_ES[0,0,:],temp_LA_ES[0,1,:],temp_LA_ES[0,2,:])
            pre =  gen_one_hot(out_LA_ES_LV[0,0,:],out_LA_ES_MYO[0,0,:],out_LA_ES_RV[0,0,:])
            vae =  gen_one_hot(out_2D_ES[0,0,:],out_2D_ES[0,1,:],out_2D_ES[0,2,:])
            pro =  gen_one_hot(proj_2d_ES[0,0,:],proj_2d_ES[0,1,:],proj_2d_ES[0,2,:])
             
            ## Dice Score for ES-LA ###

            single_LA_LV_ES = (2 * (out_LA_ES_LV * temp_LA_ES[:,0:1,:]).sum()) / (
               (out_LA_ES_LV + temp_LA_ES[:,0:1,:]).sum() + 1e-8)
           
            Dice_score_LA_LV_ES +=single_LA_LV_ES
           
            single_LA_MYO_ES = (2 * (out_LA_ES_MYO*temp_LA_ES[:,1:2,:]).sum()) / (
   (out_LA_ES_MYO + temp_LA_ES[:,1:2,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ES += single_LA_MYO_ES

            single_LA_RV_ES = (2 * (out_LA_ES_RV* temp_LA_ES[:,2:3,:]).sum()) / (
       (out_LA_ES_RV + temp_LA_ES[:,2:3,:]).sum() + 1e-8)
            Dice_score_LA_RV_ES += single_LA_RV_ES
            
                  
    Dice_RV_LA = (Dice_score_LA_RV_ES + Dice_score_LA_RV_ED)
    Dice_MYO_LA = (Dice_score_LA_MYO_ES + Dice_score_LA_MYO_ED)
    Dice_LV_LA = (Dice_score_LA_LV_ES + Dice_score_LA_LV_ED)
    
    print(f"Dice_RV_LA  : {Dice_RV_LA/len(loader)}")
    print(f"Dice_MYO_LA  : {Dice_MYO_LA/len(loader)}")
    print(f"Dice_LV_LA  : {Dice_LV_LA/len(loader)}")


    return gt,pre,vae,pro,img_LA_ES
model_1 = Autoencoder_2D2_1()
model_2 = unet_2d()


path_to_checkpoints = r"C:\My_Data\SEG.A. 2023\SDF_1.pth.tar"
path_to_checkpoints_unet_2d = r"C:\My_Data\SEG.A. 2023\CONC_2D_42.pth.tar"

def eval_():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.AdamW(model1.parameters(), betas=(0.9, 0.9),lr=0)
    checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
    model1.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    model_unet_2d = model_2.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.Adam(model_unet_2d.parameters(), betas=(0.9, 0.9),lr=0)
    checkpoint = torch.load(path_to_checkpoints_unet_2d,map_location=DEVICE)
    model_unet_2d.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    print('Dice Score for LA')
    gt,pre,vae,pro,img = check_Dice_Score(val_loader, model1,model_unet_2d, device=DEVICE)
    
    
    plt.figure()
    plt.imshow(gt)
    
    plt.figure()
    plt.imshow(pre)
    
    plt.figure()
    plt.imshow(vae)
    
    plt.figure()
    plt.imshow(pro)
    
    plt.figure()
    plt.imshow(img[0,0,:])
    
    # plt.figure()
    # plt.imshow(b[0,0,:])
    
    # plt.figure()
    # plt.imshow(c[0,0,:])
    
    # plt.figure()
    # plt.imshow(a1[0,0,:])
    
    # plt.figure()
    # plt.imshow(b1[0,0,:])
    
    # plt.figure()
    # plt.imshow(c1[0,0,:])


if __name__ == "__main__":
    eval_()
    
    
    
