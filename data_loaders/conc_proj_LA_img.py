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
from typing import List, Union, Tuple

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


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

def generate_label_3(gt):
        temp_ = np.zeros([3,gt.shape[1],gt.shape[2],gt.shape[3]])
        temp_[0:1,:,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:,:][np.where(gt==3)]=1
        return temp_

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
    
class Dataset_V(Dataset): 
    def __init__(self, img_2ds_folder,gt_2ds_folder,img_3ds_folder,gt_3ds_folder):
      
        self.img_2ds_folder = img_2ds_folder
        self.gt_2ds_folder = gt_2ds_folder
        
        self.img_3ds_folder = img_3ds_folder
        self.gt_3ds_folder = gt_3ds_folder
        
        self.img_2ds = os.listdir(img_2ds_folder)
    
    def __len__(self):
       return len(self.img_2ds)
   
    def __getitem__(self, index):
        img_2d_path = os.path.join(self.img_2ds_folder,self.img_2ds[index])
        gt_2d_path = os.path.join(self.gt_2ds_folder,self.img_2ds[index][:-12]+'.nii.gz')
        
        
        number  = self.img_2ds[index][:-18]
        img_3d_path = os.path.join(self.img_3ds_folder,number +'_SA_ES_0000.nii.gz')
        gt_3d_path = os.path.join(self.gt_3ds_folder,number +'_SA_ES.nii.gz')
        
        img_2d_itk = sitk.ReadImage(img_2d_path)
        img_2d_itk = resample_image(img_2d_itk,is_label=False) 
        img_2d = sitk.GetArrayFromImage(img_2d_itk)
        img_2d = Normalization_1(img_2d)
                      
        gt_2d_itk = sitk.ReadImage(gt_2d_path)
        gt_2d_itk = resample_image(gt_2d_itk,is_label=True) 
        gt_2d = sitk.GetArrayFromImage(gt_2d_itk)

        gt_3d_itk = sitk.ReadImage(gt_3d_path)
        
        ### projection part  #####
        
        proj_itk = SA_to_LA(gt_3d_itk,img_2d_itk)
        proj_itk = resample_image(proj_itk,is_label=True) 
        proj_itk = sitk.GetArrayFromImage(proj_itk)
        
        proj_itk = np.expand_dims(proj_itk, axis=0)
        proj_itk = generate_label_3(proj_itk) 
        proj_itk = proj_itk[:,0,:]
        
        
        print(img_2d.shape)
        print(proj_itk.shape)
        print(gt_2d.shape)
        new_img = np.concatenate((img_2d, proj_itk), axis=0)
        

        return img_2d,gt_2d,new_img, self.img_2ds[index]
        
def Data_Loader_V(img_2ds_folder,gt_2ds_folder,img_3ds_folder,gt_3ds_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(img_2ds_folder=img_2ds_folder,gt_2ds_folder=gt_2ds_folder,img_3ds_folder=img_3ds_folder,gt_3ds_folder=gt_3ds_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

img_2d_path =r'C:\My_Data\M2M Data\Dataset032_MNM\imagesTs'
gt_2d_path =r'C:\My_Data\M2M Data\Dataset032_MNM\labelsTs'

img_3d_path =r'C:\My_Data\M2M Data\Dataset031_MNM\imagesTs'
gt_3d_path =r'C:\My_Data\M2M Data\Dataset031_MNM\labelsTs'

loader = Data_Loader_V(img_2d_path,gt_2d_path,img_3d_path,gt_3d_path,1)

a = iter(loader)
#a1 = next(a)

# plt.figure()
# plt.imshow(a1[0][0,0,:])

# plt.figure()
# plt.imshow(a1[1][0,0,:])

# for i in range(4):
#     plt.figure()
#     plt.imshow(a1[2][0,i,:])

import nibabel as nib

img_paths = r'C:\My_Data\M2M Data\data\LA_proj_data\imagesTs'
gt_paths = r'C:\My_Data\M2M Data\data\LA_proj_data\labelsTs'
imgc_paths = r'C:\My_Data\M2M Data\data\LA_proj_data\imagesTc'


for i in range(1):
    a1 =next(a)
    img_name = a1[3][0][:-7]
    gt_name = img_name[:-5]


    img = a1[0][0,:].numpy()
    gt = a1[1][0,:].numpy()
    img_c = a1[2][0,:].numpy()
    
    
    print(img_c.shape)
    
        ###  saving the imgs 
    img = np.moveaxis(img, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_img = nib.Nifti1Image(img, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(img_paths,img_name + '.nii.gz'))
        
        ###  saving the gts 
    gt = np.moveaxis(gt, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_gt = nib.Nifti1Image(gt, np.eye(4))  
    to_format_gt.set_data_dtype(np.uint8)
    to_format_gt.to_filename(os.path.join(gt_paths,gt_name+ '.nii.gz'))
    
        ###  saving the gts 
    img_c = np.moveaxis(img_c, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_gt = nib.Nifti1Image(img_c, np.eye(4))  
    to_format_gt.set_data_dtype(np.uint8)
    to_format_gt.to_filename(os.path.join(imgc_paths,img_name+ '.nii.gz'))
    
    


a = sitk.ReadImage(r"C:\My_Data\M2M Data\data\LA_proj_data\imagesTc\178_LA_ED_0000.nii.gz")
a = sitk.GetArrayFromImage(a)

for i in range(4):
    plt.figure()
    plt.imshow(a[i,:])
    
    

    

a = sitk.ReadImage(r"C:\My_Data\M2M Data\data\LA_proj_data\imagesTs\178_LA_ED_0000.nii.gz")
a = sitk.GetArrayFromImage(a)

plt.figure()
plt.imshow(a[0,:])


a = sitk.ReadImage(r"C:\My_Data\M2M Data\data\LA_proj_data\labelsTs\178_LA_ED.nii.gz")
a = sitk.GetArrayFromImage(a)

plt.figure()
plt.imshow(a[0,:])



a = sitk.ReadImage(r"C:\My_Data\M2M Data\Dataset032_MNM\labelsTs\178_LA_ED.nii.gz")
a = sitk.GetArrayFromImage(a)

plt.figure()
plt.imshow(a[0,:])
