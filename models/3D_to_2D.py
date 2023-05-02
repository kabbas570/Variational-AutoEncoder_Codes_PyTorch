import torch.nn as nn
import torch

import math
class my_layer(nn.Module):
    def __init__(self,  Channel:int,Depth:int,in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight = torch.Tensor(Channel,Depth,out_features, in_features)
        self.weight = nn.Parameter(weight)
        
        # # initialize weights and biases
        ## METHOD 1
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        ##  METHOD  2
        #torch.nn.init.normal_(self.weight,mean = 0.0 ,std =1.0)  
                
    def forward(self, x):
        return x*self.weight

class Sum_Tensor(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.axis = args
    def forward(self, x):
         x = torch.sum(x, self.axis)
         return x 

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class Recons_Model22(nn.Module):
    def __init__(self):
        super(Recons_Model22, self).__init__()
        self.encoder_3d = nn.Sequential(
            my_layer(1,17,16,16),
            Sum_Tensor(2),
            nn.Sigmoid(),
            
            ## Some Extra COnvolutions ####
            Conv(1,64,1,0,1),
            nn.Sigmoid(),
            Conv(64,128,1,0,1),
            nn.Sigmoid(),
            Conv(128,64,1,0,1),
            nn.Sigmoid(),
            Conv(64,1,1,0,1),
            nn.Sigmoid(),        
        )

    def forward(self, x_3d):
        x_2 = self.encoder_3d(x_3d)
        return x_2
        
    

Input_Image_Channels = 1
def model() -> Recons_Model22:
    model = Recons_Model22()
    return model
from torchsummary import summary
model = model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(Input_Image_Channels, 17,16,16)])       
