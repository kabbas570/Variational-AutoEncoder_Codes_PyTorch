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
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.GELU(),
            
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1,2,2),(1,2,2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DoubleConv_2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_2d(in_channels, out_channels)

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

class UNet_3D_2D(nn.Module):
    def __init__(self, n_channels = 3, bilinear=False):
        super(UNet_3D_2D, self).__init__()
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
        self.outc = OutConv(32,3)
    
    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
                
        x6 = torch.sum(x6, dim = [2])
        x5 = torch.sum(x5, dim = [2])
        x4 = torch.sum(x4, dim = [2])
        x3 = torch.sum(x3, dim = [2])
        x2 = torch.sum(x2, dim = [2])
        x1 = torch.sum(x1, dim = [2])
                
        z1 = self.up0(x6, x5)
        z2 = self.up1(z1, x4)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)
        logits1 = self.outc(z5)
        
        return logits1
        
        
    
# Input_Image_Channels = 1
# def model() -> UNet_3D_2D:
#     model = UNet_3D_2D()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,17,256,256)])


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_2d(in_channels//2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_2d(in_channels//2, out_channels)

    def forward(self, x1):
        x = self.up(x1)
      
        return self.conv(x)
    

class UNet_2D_3D_no_Conc(nn.Module):
    def __init__(self, n_channels = 3, bilinear=False):
        super(UNet_2D_3D_no_Conc, self).__init__()
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
        self.outc = OutConv(32,3)
    
    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
                
        x6 = torch.sum(x6, dim = [2])
                
        z1 = self.up0(x6)
        z2 = self.up1(z1)
        z3 = self.up2(z2)
        z4 = self.up3(z3)
        z5 = self.up4(z4)
        logits1 = self.outc(z5)
        return logits1
        
       
# Input_Image_Channels = 1
# def model() -> UNet_2D_3D_no_Conc:
#     model = UNet_2D_3D_no_Conc()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,17,256,256)])


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Autoencoder_FC(nn.Module):
    def __init__(self, n_channels = 3, bilinear=False):
        super(Autoencoder_FC, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
        DoubleConv(n_channels, 32),
        Down(32, 64),
        Down(64, 128),
        Down(128, 256),
        Down(256, 512),
        Down(512, 1024 // factor),
        nn.Flatten(),
        torch.nn.Linear(8*8*1024*17, 64),
        nn.ReLU(inplace=True)
        )
        
        self.decoder =  nn.Sequential(
        torch.nn.Linear(64,8*8*1024),
        nn.ReLU(inplace=True),
        Reshape(-1,1024, 8, 8),
        Up(1024, 512 // factor, bilinear),
        Up(512, 256 // factor, bilinear),
        Up(256, 128 // factor, bilinear),
        Up(128, 64 // factor, bilinear),
        Up(64, 32, bilinear),
        OutConv(32,3),
        )
        
    def forward(self, x_in):
       x = self.encoder(x_in)
       encoded = self.decoder(x)
       return encoded


# Input_Image_Channels = 3
# def model() -> Autoencoder_FC:
#     model = Autoencoder_FC()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,17,256,256)])