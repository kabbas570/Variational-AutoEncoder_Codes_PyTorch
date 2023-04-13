import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
      

class DoubleConv_2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DoubleConv_3d(nn.Module):
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

class Down_2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class OutConv_3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)  
class Up_2d(nn.Module):
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
    
class Recons_Model(nn.Module):
    def __init__(self, n_channels = 4, bilinear=False):
        super(Recons_Model, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder_3d = nn.Sequential(
        DoubleConv_3d(n_channels, 32),
        Down_3d(32, 64),
        Down_3d(64, 128),
        Down_3d(128, 256),
        Down_3d(256, 512),
        #Down_3d(512, 1024 // factor),
        nn.Flatten(),
        torch.nn.Linear(16*16*512*17, 64),
        #nn.ReLU(inplace=True)
        )
        
        self.decoder_2d =  nn.Sequential(
        torch.nn.Linear(64,16*16*512),
        #nn.ReLU(inplace=True),
        Reshape(-1,512, 16,16),
        #Up_2d(1024, 512 // factor, bilinear),
        Up_2d(512, 256 // factor, bilinear),
        Up_2d(256, 128 // factor, bilinear),
        Up_2d(128, 64 // factor, bilinear),
        Up_2d(64, 32, bilinear),
        OutConv_2d(32,4),
        )
        
        self.encoder_2d = nn.Sequential(
        DoubleConv_2d(n_channels, 32),
        Down_2d(32, 64),
        Down_2d(64, 128),
        Down_2d(128, 256),
        Down_2d(256, 512),
        #Down_2d(512, 1024 // factor),
        nn.Flatten(),
        torch.nn.Linear(16*16*512,64),
        #nn.ReLU(inplace=True)
        )
        
        self.decoder_3d =  nn.Sequential(
        torch.nn.Linear(64,16*16*512*17),
        #nn.ReLU(inplace=True),
        Reshape(-1,512,17,16,16),
        #Up_3d(1024, 512 // factor, bilinear),
        Up_3d(512, 256 // factor, bilinear),
        Up_3d(256, 128 // factor, bilinear),
        Up_3d(128, 64 // factor, bilinear),
        Up_3d(64, 32, bilinear),
        OutConv_3d(32,4),
        )
        
        self.proj_2d_to_3d = nn.Sequential(
        torch.nn.Linear(64,64),
        #nn.ReLU(inplace=True)
        )
        
        self.proj_3d_to_2d = nn.Sequential(
        torch.nn.Linear(64,64),
        #nn.ReLU(inplace=True)
        )
        
    def forward(self, x_2d,x_3d):
        
        x_2 = self.encoder_2d(x_2d)
        x_2_projected = self.proj_2d_to_3d(x_2)
        encoded_3d = self.decoder_3d(x_2_projected)
        
        x_3 = self.encoder_3d(x_3d)
        x_3_projected = self.proj_3d_to_2d(x_3)
        encoded_2d = self.decoder_2d(x_3_projected)
       
        return encoded_2d,encoded_3d,x_2,x_3,x_2_projected,x_3_projected


# Input_Image_Channels = 4
# def model() -> Autoencoder_FC:
#     model = Autoencoder_FC()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256),(Input_Image_Channels,17,256,256)])
