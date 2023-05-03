'''import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch     

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class DoubleConv(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Sigmoid(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
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
            self.conv = DoubleConv(in_channels//2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
class DoubleConv_3d(nn.Module):
    """(convolution => [BN] ) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.Sigmoid(),
            
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_3d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1,2,2),(1,2,2)),
            DoubleConv_3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv_3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Up_3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_3d(in_channels//2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv_3d(in_channels//2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class Autoencoder_2D2(nn.Module):
    def __init__(self, n_channels = 4, bilinear=False):
        super(Autoencoder_2D2, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
        DoubleConv(n_channels, 32),
        Down(32, 64),
        Down(64, 128),
        Down(128, 256),
        Down(256, 512),
        #Down(512, 1024 // factor),
        )
        self.decoder =  nn.Sequential(
        #Up(1024, 512 // factor, bilinear),
        Up(512, 256 // factor, bilinear),
        Up(256, 128 // factor, bilinear),
        Up(128, 64 // factor, bilinear),
        Up(64, 32, bilinear),
        OutConv(32,4),
        )
    def forward(self, x_in):
       x = self.encoder(x_in)
       encoded = self.decoder(x)
       return encoded
   
# Input_Image_Channels = 4
# def model() -> Autoencoder_2D2:
#     model = Autoencoder_2D2()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])
    
class Autoencoder_3D3(nn.Module):
    def __init__(self, n_channels = 4, bilinear=False):
        super(Autoencoder_3D3, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
        DoubleConv_3d(n_channels, 32),
        Down_3d(32, 64),
        Down_3d(64, 128),
        Down_3d(128, 256),
        Down_3d(256, 512),
        #Down_3d(512, 1024 // factor),
        )
        self.decoder =  nn.Sequential(
        #
        #Up_3d(1024, 512 // factor, bilinear),
        Up_3d(512, 256 // factor, bilinear),
        Up_3d(256, 128 // factor, bilinear),
        Up_3d(128, 64 // factor, bilinear),
        Up_3d(64, 32, bilinear),
        OutConv_3d(32,4),
        )
    def forward(self, x_in):
       x = self.encoder(x_in)
       encoded = self.decoder(x)
       return encoded
   
    
# Input_Image_Channels = 4
# def model() -> Autoencoder_3D3:
#     model = Autoencoder_3D3()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,17,256,256)])



class Recons_Model_1(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_1, self).__init__()

        self.encoder_3d = nn.Sequential(
        nn.Flatten(),
        torch.nn.Linear(16*16*512*17, 512),
        nn.BatchNorm1d(512),
        nn.Sigmoid(),
        torch.nn.Linear(512,256),
        nn.BatchNorm1d(256),
        nn.Sigmoid()
        )
        
        self.decoder_2d =  nn.Sequential(
        torch.nn.Linear(256,16*16*512),
        nn.BatchNorm1d(16*16*512),
        nn.Sigmoid(),
        Reshape(-1,512,16,16)
        )
  
    def forward(self, x_3d):
        x_3 = self.encoder_3d(x_3d)
        encoded_2d = self.decoder_2d(x_3)
        return encoded_2d
        
        
# def model() -> Recons_Model_1:
#     model = Recons_Model_1()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(512,17,16,16)])

class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels//2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        
        return self.conv(x1) 
        
class Reshape1(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
        return torch.sum(x,dim =2)
        
class Recons_Model_2(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_2, self).__init__()
        
        self.encoder_3d = nn.Sequential(
            Reshape1(),
            Down(512,768),
            Down(768,1024)
        )
        self.decoder_2d =  nn.Sequential(
        Up1(1024, 768),
        Up1(768, 512)
        )
  
    def forward(self, x_3d):
        x_3 = self.encoder_3d(x_3d)
        encoded_2d = self.decoder_2d(x_3)
        return encoded_2d     
        

class Conv_11(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels,stride,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=0,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Recons_Model_3(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_3, self).__init__()
        
        self.encoder_3d = nn.Sequential(
            Reshape1(),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
        )

    def forward(self, x_3d):
        x_2 = self.encoder_3d(x_3d)
        return x_2     
        

class Conv_17(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels,stride,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding='same',stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Recons_Model_4(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_4, self).__init__()
        
        self.encoder_3d = nn.Sequential(
            Reshape1(),
            Conv_17(16,16,1,7),
            Conv_17(16,16,1,5),
            Conv_17(16,16,1,3),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
            Conv_11(16,16,1,1),
        )

    def forward(self, x_3d):
        x_2 = self.encoder_3d(x_3d)
        return x_2 
        
        

class Move_Axis(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.moveaxis(x,2,4)  ## --->  [b,c,d,h,w] --> [b,c,h,w,d] 
         return x

class SeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.squeeze(x, 4)
         return x
         
         
class Recons_Model_5(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_5, self).__init__()
        
        self.encoder_3d = nn.Sequential(
        Move_Axis(),
        torch.nn.Linear(17,1),
        SeQ(),
        #nn.LayerNorm(16),
        nn.InstanceNorm2d(16),
        nn.ReLU(inplace=True),
        
        Conv_17(16,16,1,7),
        Conv_17(16,16,1,5),
        Conv_17(16,16,1,3),
        Conv_11(16,16,1,1),
        Conv_11(16,16,1,1),
        Conv_11(16,16,1,1),
        )

    def forward(self, x_3d):
        x_2 = self.encoder_3d(x_3d)
        return x_2 
        
        
         
class Recons_Model_6(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_6, self).__init__()
        
        self.encoder_3d = nn.Sequential(
        Move_Axis(),
        
        torch.nn.Linear(17,64),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),
        
        torch.nn.Linear(64,128),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        
        
        torch.nn.Linear(128,256),
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),
        
        torch.nn.Linear(256,16),
        nn.InstanceNorm2d(16),
        nn.ReLU(inplace=True),
        
        torch.nn.Linear(256,1),
        nn.InstanceNorm2d(16),
        nn.ReLU(inplace=True),
        
        SeQ(),
        
        Conv_17(16,16,1,7),
        Conv_17(16,16,1,5),
        Conv_17(16,16,1,3),
        Conv_11(16,16,1,1),
        Conv_11(16,16,1,1),
        Conv_11(16,16,1,1),
        )

    def forward(self, x_3d):
        x_2 = self.encoder_3d(x_3d)
        return x_2 '''
        

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch 


class Move_Axis(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.moveaxis(x,2,4)  ## --->  [b,c,d,h,w] --> [b,c,h,w,d] 
         return x

class SeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.squeeze(x, 4)
         return x
     
class Conv_11(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels,stride,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=0,stride=stride),
            #nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Conv_17(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels,stride,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding='same',stride=stride),
            #nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)     

class Recons_Model_6(nn.Module):
    def __init__(self,  bilinear=False):
        super(Recons_Model_6, self).__init__()
        
        self.encoder_3d = nn.Sequential(
        Move_Axis(),
        
        torch.nn.Linear(17,64),
        #nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),
        
        
        torch.nn.Linear(64,128),
        #nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        
        
        torch.nn.Linear(128,256),
        #nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),
        
        torch.nn.Linear(256,16),
        #nn.InstanceNorm2d(16),
        nn.ReLU(inplace=True),
        
        torch.nn.Linear(16,1),
        #nn.InstanceNorm2d(16),
        nn.ReLU(inplace=True),
        
        SeQ(),
        #nn.InstanceNorm2d(16),
        
        Conv_17(16,16,1,7),
        Conv_17(16,16,1,5),
        Conv_17(16,16,1,3),
        Conv_11(16,16,1,1),
        Conv_11(16,16,1,1),
        Conv_11(16,16,1,1),
        
        
        )

    def forward(self, x_3d):
        x_2 = self.encoder_3d(x_3d)
        return x_2 
    

# Input_Image_Channels = 16
# def model() -> Recons_Model_6:
#     model = Recons_Model_6()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 17,16,16)])