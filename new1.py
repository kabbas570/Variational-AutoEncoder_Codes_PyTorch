from tqdm import tqdm
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


#### Model ###
class module_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        max_result,_ = torch.max(x,dim=1,keepdim=True)
        avg_result = torch.mean(x,dim=1,keepdim=True)
        std_result = torch.std(x,dim=1,keepdim=True)
        min_result,_ = torch.min(x,dim=1,keepdim=True)
        result = torch.cat([max_result,std_result,avg_result,min_result],1)
        return result

class module_2(nn.Module):
    def __init__(self,in_channels, out_channels,):
        super().__init__()
        
        self.features = module_1()
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
            
    def forward(self,x):
        
        feature = self.features(x)
        c3 = self.conv_3x3(feature)
        c5 = self.conv_5x5(feature)
        c7 = self.conv_7x7(feature)
        
        x = torch.cat([c3,c5,c7,feature], dim=1)

        return x
    
    
def double_conv(in_channels, out_channels,stride=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

f_dim = 16
class proposed1(nn.Module):

    def __init__(self, n_class=10):
        super().__init__()
        
        self.Block1_conv = double_conv(1,f_dim)
        self.Block2_conv = double_conv(4,f_dim)
        self.Block3_conv = double_conv(4,f_dim)
        self.Block4_conv = double_conv(4,f_dim)
        self.Block5_conv = double_conv(4,f_dim)
        self.module1 = module_2(4,16)


        self.pooling=nn.AdaptiveAvgPool2d((1,1))# 256 x 1 x 1
        self.fc1 = nn.Linear(f_dim, f_dim)
        self.fc2 = nn.Linear(f_dim, 10)
        
    def forward(self,x1):
        
        conv1 = self.Block1_conv(x1)
        conv1 = self.module1(conv1)
        
        conv2 = self.Block2_conv(conv1)
        conv2 = self.module1(conv2)
        
        conv3 = self.Block3_conv(conv2)
        conv3 = self.module1(conv3)
        
        conv4 = self.Block4_conv(conv3)
        conv4 = self.module1(conv4)
        
        conv5 = self.Block5_conv(conv4)
        
        pool_ = self.pooling(conv5)
        
        pool_ = pool_.reshape(pool_.shape[0], -1)
        
        x = F.relu(self.fc1(pool_))
        x = F.relu(self.fc2(x))
        return x

Input_Image_Channels = 1
def model() -> proposed1:
    model = proposed1()
    return model
from torchsummary import summary
model = model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(Input_Image_Channels, 256,256)])
