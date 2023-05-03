##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##import torchvision
##            
##class DoubleConv(nn.Module):
##    """(convolution => [BN] => ReLU) * 2"""
##
##    def __init__(self, in_channels, out_channels, mid_channels=None):
##        super().__init__()
##        if not mid_channels:
##            mid_channels = out_channels
##        self.double_conv = nn.Sequential(
##            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
##            nn.BatchNorm2d(mid_channels),
##            nn.ReLU(inplace=True),
##
##            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
##            nn.BatchNorm2d(out_channels),
##            nn.ReLU(inplace=True),
##        )
##
##    def forward(self, x):
##        return self.double_conv(x)
##
##class Down(nn.Module):
##    """Downscaling with maxpool then double conv"""
##
##    def __init__(self, in_channels, out_channels):
##        super().__init__()
##        self.maxpool_conv = nn.Sequential(
##            nn.MaxPool2d(2),
##            DoubleConv(in_channels, out_channels)
##        )
##
##    def forward(self, x):
##        return self.maxpool_conv(x)
##
##class Up(nn.Module):
##    """Upscaling then double conv"""
##
##    def __init__(self, in_channels, out_channels, bilinear=True):
##        super().__init__()
##
##        # if bilinear, use the normal convolutions to reduce the number of channels
##        if bilinear:
##            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
##            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
##        else:
##            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
##            self.conv = DoubleConv(in_channels//2, out_channels)
##
##    def forward(self, x1):
##        x1 = self.up(x1)
##        
##        return self.conv(x1)
##    
##class OutConv(nn.Module):
##    def __init__(self, in_channels, out_channels):
##        super(OutConv, self).__init__()
##        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
##
##    def forward(self, x):
##        return self.conv(x)
##
##class UNet_noSkip(nn.Module):
##    def __init__(self, n_channels = 1, bilinear=False):
##        super(UNet_noSkip, self).__init__()
##        self.n_channels = n_channels
##        self.bilinear = bilinear
##
##        self.inc = DoubleConv(n_channels, 32)
##        self.down1 = Down(32, 64)
##        self.down2 = Down(64, 128)
##        self.down3 = Down(128, 256)
##        self.down4 = Down(256, 512)
##        factor = 2 if bilinear else 1
##        self.down5 = Down(512, 1024 // factor)
##        
##        self.up0 = Up(1024, 512 // factor, bilinear)
##        self.up1 = Up(512, 256 // factor, bilinear)
##        self.up2 = Up(256, 128 // factor, bilinear)
##        self.up3 = Up(128, 64 // factor, bilinear)
##        self.up4 = Up(64, 32, bilinear)
##        self.outc = OutConv(32,1)
##                                
##        self.act = nn.Sigmoid()
##        
##    def forward(self, x):
##
##        x1 = self.inc(x)
##        x2 = self.down1(x1)
##        x3 = self.down2(x2)
##        x4 = self.down3(x3)
##        x5 = self.down4(x4)
##        x6 = self.down5(x5)
##        
##        z1 = self.up0(x6)
##        z2 = self.up1(z1)
##        z3 = self.up2(z2)
##        z4 = self.up3(z3)
##        z5 = self.up4(z4)
##        logits1 = self.outc(z5)
##         
##        return self.act(logits1)
##        
### Input_Image_Channels = 1
### def model() -> UNet_noSkip:
###     model = UNet_noSkip()
###     return model
### from torchsummary import summary
### model = model()
### DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
### model.to(device=DEVICE,dtype=torch.float)
### summary(model, [(Input_Image_Channels, 256,256)])
##
##class UNet_noSkip1(nn.Module):
##    def __init__(self, n_channels = 1, bilinear=False):
##        super(UNet_noSkip1, self).__init__()
##        self.n_channels = n_channels
##        self.bilinear = bilinear
##
##        factor = 2 if bilinear else 1
##        
##        self.encoder = nn.Sequential(
##        DoubleConv(n_channels, 32),
##        Down(32, 64),
##        Down(64, 128),
##        Down(128, 256),
##        Down(256, 512),
##        Down(512, 1024 // factor)
##        )
##        
##        self.decoder =  nn.Sequential(
##            
##        Up(1024, 512 // factor, bilinear),
##        Up(512, 256 // factor, bilinear),
##        Up(256, 128 // factor, bilinear),
##        Up(128, 64 // factor, bilinear),
##        Up(64, 32, bilinear),
##        OutConv(32,1),
##        nn.Sigmoid()
##        )
##        
##    def forward(self, x_in):
##        
##       x = self.encoder(x_in)
##       encoded = self.decoder(x)
##       
##       return encoded
##       
##
##class Reshape(nn.Module):
##    def __init__(self, *args):
##        super().__init__()
##        self.shape = args
##
##    def forward(self, x):
##        return x.view(self.shape)
##    
##Latent_Size = 64
##class UNet_noSkip2(nn.Module):
##    def __init__(self, n_channels = 1, bilinear=False):
##        super(UNet_noSkip2, self).__init__()
##        self.n_channels = n_channels
##        self.bilinear = bilinear
##
##        factor = 2 if bilinear else 1
##        
##        self.encoder = nn.Sequential(
##        DoubleConv(n_channels, 32),
##        Down(32, 64),
##        Down(64, 128),
##        Down(128, 256),
##        Down(256, 512),
##        Down(512, 1024 // factor),
##        nn.Flatten(),
##        torch.nn.Linear(8*8*1024, Latent_Size),
##        nn.ReLU(inplace=True)
##        )
##        
##        self.decoder =  nn.Sequential(
##        
##        torch.nn.Linear(Latent_Size,8*8*1024),
##        nn.ReLU(inplace=True),
##        Reshape(-1, 1024, 8, 8),
##        Up(1024, 512 // factor, bilinear),
##        Up(512, 256 // factor, bilinear),
##        Up(256, 128 // factor, bilinear),
##        Up(128, 64 // factor, bilinear),
##        Up(64, 32, bilinear),
##        OutConv(32,1),
##        nn.Sigmoid()
##        )
##        
##    def forward(self, x_in):
##        
##       x = self.encoder(x_in)
##       encoded = self.decoder(x)
##       
##       return encoded
#
#
#
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#            
#class DoubleConv(nn.Module):
#    """(convolution => [BN] => ReLU) * 2"""
#
#    def __init__(self, in_channels, out_channels, mid_channels=None):
#        super().__init__()
#        if not mid_channels:
#            mid_channels = out_channels
#        self.double_conv = nn.Sequential(
#            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(mid_channels),
#            nn.ReLU(inplace=True),
#
#            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True),
#        )
#
#    def forward(self, x):
#        return self.double_conv(x)
#
#class Down(nn.Module):
#    """Downscaling with maxpool then double conv"""
#
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.maxpool_conv = nn.Sequential(
#            nn.MaxPool2d(2),
#            DoubleConv(in_channels, out_channels)
#        )
#
#    def forward(self, x):
#        return self.maxpool_conv(x)
#
#class Up(nn.Module):
#    """Upscaling then double conv"""
#
#    def __init__(self, in_channels, out_channels, bilinear=True):
#        super().__init__()
#
#        # if bilinear, use the normal convolutions to reduce the number of channels
#        if bilinear:
#            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#            self.conv = DoubleConv(in_channels//2, out_channels, in_channels // 2)
#        else:
#            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#            self.conv = DoubleConv(in_channels//2, out_channels)
#
#    def forward(self, x1):
#        x1 = self.up(x1)
#        
#        return self.conv(x1)
#
#class OutConv(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super(OutConv, self).__init__()
#        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#    def forward(self, x):
#        return self.conv(x)
#        
#class Autoencoder_2D2(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_2D2, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        Down(256, 512),
#        #Down(512, 1024 // factor),
#        )
#        self.decoder =  nn.Sequential(
#        #Up(1024, 512 // factor, bilinear),
#        Up(512, 256 // factor, bilinear),
#        Up(256, 128 // factor, bilinear),
#        Up(128, 64 // factor, bilinear),
#        Up(64, 32, bilinear),
#        OutConv(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       return encoded
#   
##Input_Image_Channels = 1
##def model() -> UNet_noSkip2:
##    model = UNet_noSkip2()
##    return model
##from torchsummary import summary
##model = model()
##DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
##model.to(device=DEVICE,dtype=torch.float)
##summary(model, [(Input_Image_Channels, 256,256)])
#
#class Segmentar(nn.Module):
#    def __init__(self, n_channels = 1, bilinear=False):
#        super(Segmentar, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        Down(256, 512),
#        Down(512, 1024 // factor),
#        nn.Flatten(),
#        torch.nn.Linear(8*8*1024, 64),
#        nn.Sigmoid()
#        )
#           
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       return x
#       
#class Reshape(nn.Module):
#    def __init__(self, *args):
#        super().__init__()
#        self.shape = args
#
#    def forward(self, x):
#        return x.view(self.shape)
#        
#class Autoencoder_FC(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_FC, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        Down(256, 512),
#        Down(512, 1024 // factor),
#        nn.Flatten(),
#        torch.nn.Linear(8*8*1024, 128),
#        nn.Sigmoid()
#        )
#        
#        self.decoder =  nn.Sequential(
#        torch.nn.Linear(128,8*8*1024),
#        Reshape(-1, 1024, 8, 8),
#        Up(1024, 512 // factor, bilinear),
#        Up(512, 256 // factor, bilinear),
#        Up(256, 128 // factor, bilinear),
#        Up(128, 64 // factor, bilinear),
#        Up(64, 32, bilinear),
#        OutConv(32,4),
#        nn.Sigmoid()
#        )
#        
#    def forward(self, x_in):
#        
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       
#       return encoded
#       
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#
#class DoubleConv_3d(nn.Module):
#    """(convolution => [BN] => ReLU) * 2"""
#
#    def __init__(self, in_channels, out_channels, mid_channels=None):
#        super().__init__()
#        if not mid_channels:
#            mid_channels = out_channels
#        self.double_conv = nn.Sequential(
#            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(mid_channels),
#            nn.GELU(),
#            
#            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(out_channels),
#            nn.GELU(),
#        )
#
#    def forward(self, x):
#        return self.double_conv(x)
#
#class Down_3d(nn.Module):
#    """Downscaling with maxpool then double conv"""
#
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.maxpool_conv = nn.Sequential(
#            nn.MaxPool3d((1,2,2),(1,2,2)),
#            DoubleConv_3d(in_channels, out_channels)
#        )
#
#    def forward(self, x):
#        return self.maxpool_conv(x)
#
#class OutConv_3d(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super(OutConv_3d, self).__init__()
#        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#
#    def forward(self, x):
#        return self.conv(x)
#
#class Up_3d(nn.Module):
#    """Upscaling then double conv"""
#
#    def __init__(self, in_channels, out_channels, bilinear=True):
#        super().__init__()
#
#        # if bilinear, use the normal convolutions to reduce the number of channels
#        if bilinear:
#            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#            self.conv = DoubleConv_3d(in_channels//2, out_channels, in_channels // 2)
#        else:
#            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
#            self.conv = DoubleConv_3d(in_channels//2, out_channels)
#
#    def forward(self, x1):
#        x1 = self.up(x1)
#        return self.conv(x1)
#    
#class Autoencoder_3D3(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_3D3, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv_3d(n_channels, 32),
#        Down_3d(32, 64),
#        Down_3d(64, 128),
#        Down_3d(128, 256),
#        Down_3d(256, 512),
#        #Down_3d(512, 1024 // factor),
#        )
#        self.decoder =  nn.Sequential(
#        #
#        #Up_3d(1024, 512 // factor, bilinear),
#        Up_3d(512, 256 // factor, bilinear),
#        Up_3d(256, 128 // factor, bilinear),
#        Up_3d(128, 64 // factor, bilinear),
#        Up_3d(64, 32, bilinear),
#        OutConv_3d(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       return encoded
#   
#    
## Input_Image_Channels = 4
## def model() -> Autoencoder_3D3:
##     model = Autoencoder_3D3()
##     return model
## DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
## from torchsummary import summary
## model = model()
## model.to(device=DEVICE,dtype=torch.float)
## summary(model, [(Input_Image_Channels,17,256,256)])
#
#
#
#class Recons_Model(nn.Module):
#    def __init__(self,  bilinear=False):
#        super(Recons_Model, self).__init__()
#
#        self.encoder_3d = nn.Sequential(
#        nn.Flatten(),
#        torch.nn.Linear(16*16*512*17, 1024),
#        nn.ReLU(inplace=True),
#        torch.nn.Linear(1024,512),
#        nn.ReLU(inplace=True)
#        )
#        self.encoder_2d = nn.Sequential(
#        nn.Flatten(),
#        torch.nn.Linear(16*16*512, 1024),
#        nn.ReLU(inplace=True),
#        torch.nn.Linear(1024,512),
#        nn.ReLU(inplace=True)
#        )
#        
#        self.decoder_2d =  nn.Sequential(
#        torch.nn.Linear(512,16*16*512),
#        nn.ReLU(inplace=True),
#        Reshape(-1,512,16,16)
#        )
#         
#        self.decoder_3d =  nn.Sequential(
#        torch.nn.Linear(512,16*16*512*17),
#        nn.ReLU(inplace=True),
#        Reshape(-1,512,17,16,16)
#        )
#        
#        
#    def forward(self, x_2d,x_3d):
#        x_2 = self.encoder_2d(x_2d)
#        encoded_3d = self.decoder_3d(x_2)
#        x_3 = self.encoder_3d(x_3d)
#        encoded_2d = self.decoder_2d(x_3)
#        return encoded_2d,encoded_3d
#
#
## def model() -> Recons_Model:
##     model = Recons_Model()
##     return model
## DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
## from torchsummary import summary
## model = model()
## model.to(device=DEVICE,dtype=torch.float)
## summary(model, [(512, 16,16),(512,17,16,16)])
#
#
#class Autoencoder_3D3_1(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_3D3_1, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv_3d(n_channels, 32),
#        Down_3d(32, 64),
#        Down_3d(64, 128),
#        Down_3d(128, 256),
#        Down_3d(256, 512),
#        )
#        self.L_Space = nn.Sequential(
#        Up_3d(512, 256 // factor, bilinear)
#        )
#        self.decoder =  nn.Sequential(
#        Up_3d(256, 128 // factor, bilinear),
#        Up_3d(128, 64 // factor, bilinear),
#        Up_3d(64, 32, bilinear),
#        OutConv_3d(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       x_L = self.L_Space(x)
#       encoded = self.decoder(x_L)
#       return encoded
#   
##Input_Image_Channels = 4
##def model() -> Autoencoder_3D3_1:
##    model = Autoencoder_3D3_1()
##    return model
##DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
##from torchsummary import summary
##model = model()
##model.to(device=DEVICE,dtype=torch.float)
##summary(model, [(Input_Image_Channels,17,256,256)])
#
#class Autoencoder_2D2_1(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_2D2_1, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        Down(256, 512),
#        )
#        self.L_Space = nn.Sequential(
#        Up(512, 256 // factor, bilinear)
#        )
#        self.decoder =  nn.Sequential(
#        Up(256, 128 // factor, bilinear),
#        Up(128, 64 // factor, bilinear),
#        Up(64, 32, bilinear),
#        OutConv(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       x_L = self.L_Space(x)
#       encoded = self.decoder(x_L)
#       return encoded
#   
## Input_Image_Channels = 4
## def model() -> Autoencoder_2D2_1:
##     model = Autoencoder_2D2_1()
##     return model
## from torchsummary import summary
## model = model()
## DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
## model.to(device=DEVICE,dtype=torch.float)
## summary(model, [(Input_Image_Channels, 256,256)])
#
#
#BN_Feature = 32
#class Autoencoder_2D2_2(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_2D2_2, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        Down(256, 512),
#        nn.Flatten(),
#        torch.nn.Linear(16*16*512, 2*BN_Feature),
#        #nn.BatchNorm1d(2*BN_Feature),
#        #nn.ReLU(inplace=True),
#        #torch.nn.Linear(2*BN_Feature, BN_Feature),
#        #nn.BatchNorm1d(BN_Feature),
#        # nn.ReLU(inplace=True),
#        nn.Sigmoid()
#        )
#        
#        self.decoder =  nn.Sequential(
#        #torch.nn.Linear(BN_Feature, 2*BN_Feature),
#        #nn.BatchNorm1d(2*BN_Feature),
#        #nn.ReLU(inplace=True),
#        torch.nn.Linear(2*BN_Feature,16*16*512),
#        #nn.BatchNorm1d(16*16*512),
#        nn.ReLU(inplace=True),
#        Reshape(-1,512,16,16),
#        Up(512, 256 // factor, bilinear),
#        Up(256, 128 // factor, bilinear),
#        Up(128, 64 // factor, bilinear),
#        Up(64, 32, bilinear),
#        OutConv(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       return encoded
#   
## Input_Image_Channels = 4
## def model() -> Autoencoder_2D2_2:
##     model = Autoencoder_2D2_2()
##     return model
## from torchsummary import summary
## model = model()
## DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
## model.to(device=DEVICE,dtype=torch.float)
## summary(model, [(Input_Image_Channels, 256,256)])
#
#
#
#class Autoencoder_3D3_2(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_3D3_2, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv_3d(n_channels, 32),
#        Down_3d(32, 64),
#        Down_3d(64, 128),
#        Down_3d(128, 256),
#        Down_3d(256, 512),
#        nn.Flatten(),
#        torch.nn.Linear(17*16*16*512, BN_Feature),
#        nn.ReLU(inplace=True)
#        )
#
#        self.decoder =  nn.Sequential(
#        torch.nn.Linear(BN_Feature,17*16*16*512),
#        nn.ReLU(inplace=True),
#        Reshape(-1,512,17,16,16),
#        Up_3d(512, 256 // factor, bilinear),
#        Up_3d(256, 128 // factor, bilinear),
#        Up_3d(128, 64 // factor, bilinear),
#        Up_3d(64, 32, bilinear),
#        OutConv_3d(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       return encoded
#   
## Input_Image_Channels = 4
## def model() -> Autoencoder_3D3_2:
##     model = Autoencoder_3D3_2()
##     return model
## DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
## from torchsummary import summary
## model = model()
## model.to(device=DEVICE,dtype=torch.float)
## summary(model, [(Input_Image_Channels,17,256,256)])



import torch
import torch.nn as nn
            
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
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

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    



#class Autoencoder_2D2_2(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_2D2_2, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        Down(256, 512),
#        nn.Flatten(),
#        torch.nn.Linear(16*16*512, 2*BN_Feature),
#        nn.BatchNorm1d(2*BN_Feature),
#        nn.Sigmoid()
#        )
#        
#        self.decoder =  nn.Sequential(
#        torch.nn.Linear(2*BN_Feature,16*16*512),
#        nn.BatchNorm1d(16*16*512),
#        nn.Sigmoid(),
#        Reshape(-1,512,16,16),
#        Up(512, 256 // factor, bilinear),
#        Up(256, 128 // factor, bilinear),
#        Up(128, 64 // factor, bilinear),
#        Up(64, 32, bilinear),
#        OutConv(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       return encoded
   
# Input_Image_Channels = 4
# def model() -> Autoencoder_2D2_2:
#     model = Autoencoder_2D2_2()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])


BN_Feature = 64  
class Autoencoder_2D2_2(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(Autoencoder_2D2_2, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
        DoubleConv(n_channels, 16),
        Down(16, 32),
        Down(32, 64),
        Down(64, 128),
        Down(128, 256),
        Down(256, 512),
        nn.Dropout(p=0.2),
        nn.Flatten(),
        torch.nn.Linear(8*8*512, 4096),
        nn.Dropout(p=0.2),
        nn.InstanceNorm1d(4096),
        nn.ReLU(),
        torch.nn.Linear(4096,BN_Feature),
        # nn.InstanceNorm1d(BN_Feature),
        nn.ReLU()
        )
        
        self.decoder =  nn.Sequential(
        torch.nn.Linear(BN_Feature,4096),
        nn.InstanceNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        torch.nn.Linear(4096,8*8*512),
        nn.InstanceNorm1d(8*8*512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        Reshape(-1,512,8,8),
        Up(512, 256 // factor, bilinear),
        Up(256, 128 // factor, bilinear),
        Up(128, 64 // factor, bilinear),
        Up(64, 32, bilinear),
        Up(32, 16, bilinear),
        OutConv(16,1),
        )
    def forward(self, x_in):
       x = self.encoder(x_in)
       encoded = self.decoder(x)
       return encoded
       
       
#class Autoencoder_2D2_3(nn.Module):
#    def __init__(self, n_channels = 4, bilinear=False):
#        super(Autoencoder_2D2_3, self).__init__()
#        self.n_channels = n_channels
#        self.bilinear = bilinear
#
#        factor = 2 if bilinear else 1
#        
#        self.encoder = nn.Sequential(
#        DoubleConv(n_channels, 16),
#        Down(16, 32),
#        Down(32, 64),
#        Down(64, 128),
#        Down(128, 256),
#        OutConv(256,8),
#
#        )
#        
#        self.decoder =  nn.Sequential(
#        Up(8, 256 // factor, bilinear),
#        Up(256, 128 // factor, bilinear),
#        Up(128, 64 // factor, bilinear),
#        Up(64, 32, bilinear),
#        OutConv(32,4),
#        )
#    def forward(self, x_in):
#       x = self.encoder(x_in)
#       encoded = self.decoder(x)
#       return encoded
       
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DoubleConv_3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.GELU(),
            
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
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

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
BN_Feature = 64
mid_features = 512
class Autoencoder_3D3_1(nn.Module):
    def __init__(self, n_channels = 4, bilinear=False):
        super(Autoencoder_3D3_1, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
        DoubleConv_3d(n_channels, 16),
        Down_3d(16,32),
        Down_3d(32,64),
        Down_3d(64,128),
        Down_3d(128,256),
        Down_3d(256, 512),
        nn.Flatten(),
        torch.nn.Linear(17*8*8*512, mid_features),
        nn.InstanceNorm1d(mid_features),
        nn.ReLU(),
        torch.nn.Linear(mid_features,BN_Feature),
        nn.ReLU()
        )

        self.decoder =  nn.Sequential(
        torch.nn.Linear(BN_Feature,mid_features),
        nn.InstanceNorm1d(mid_features),
        nn.ReLU(),
        torch.nn.Linear(mid_features,17*8*8*512),
        nn.InstanceNorm1d(17*8*8*512),
        nn.ReLU(),
        Reshape(-1,512,17,8,8),
        Up_3d(512, 256 // factor, bilinear),
        Up_3d(256, 128 // factor, bilinear),
        Up_3d(128, 64 // factor, bilinear),
        Up_3d(64, 32, bilinear),
        Up_3d(32, 16, bilinear),
        OutConv_3d(16,4),
        )
    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded
  
#Input_Image_Channels = 4
#def model() -> Autoencoder_3D3_1:
#    model = Autoencoder_3D3_1()
#    return model
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#from torchsummary import summary
#model = model()
#model.to(device=DEVICE,dtype=torch.float)
#summary(model, [(Input_Image_Channels,17,256,256)])