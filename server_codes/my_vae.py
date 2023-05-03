import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class double_conv01(nn.Module):
    def __init__(self, in_channels, out_channels,f_size):
        super(double_conv01, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv11(nn.Module):
    def __init__(self, in_channels, out_channels, f_size,p_size,stride=2):
        super(double_conv11, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv_u1(nn.Module):
    def __init__(self, in_channels, out_channels,f_size,st_size):
        super(double_conv_u1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            ### up-sapmling part
            nn.ConvTranspose2d(out_channels,out_channels, kernel_size=2, stride=st_size),
            nn.ReLU(inplace=True),
        ) 
    def forward(self, x):
        return self.conv(x)
    
def double_conv_u1_last(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

Base =  32
Latent_Size = 64
           
class VAE_1(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()
        

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        nn.Flatten()
        )
        
        self.z_mean = torch.nn.Linear(8*8*32*Base, Latent_Size)
        self.z_log_var = torch.nn.Linear(8*8*32*Base, Latent_Size)
        self.reshape1 = torch.nn.Linear(Latent_Size, 8*8*32*Base)
                        
        self.decoder = nn.Sequential(
                torch.nn.Linear(64,8*8*32*Base),
                Reshape(-1, 512, 8, 8),
                double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(32*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, 1),
                torch.nn.Sigmoid()
                )
        
    def reparameterize(self, z_mu, z_log_var):
                eps = torch.randn(z_mu.size(0), z_mu.size(1))
                eps = eps.to(device=DEVICE,dtype=torch.float) 
                z = z_mu + eps * torch.exp(z_log_var/2.).to(device=DEVICE,dtype=torch.float) 
                return z
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)  
        encoded = self.decoder(encoded)
        
        return encoded, z_mean, z_log_var
    
# Input_Image_Channels = 1
# def model() -> VAE_1:
#     model = VAE_1()
#     return model
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])


import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class double_conv01(nn.Module):
    def __init__(self, in_channels, out_channels,f_size):
        super(double_conv01, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv11(nn.Module):
    def __init__(self, in_channels, out_channels, f_size,p_size,stride=2):
        super(double_conv11, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv_u1(nn.Module):
    def __init__(self, in_channels, out_channels,f_size,st_size):
        super(double_conv_u1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            ### up-sapmling part
            nn.ConvTranspose2d(out_channels,out_channels, kernel_size=2, stride=st_size),
            nn.ReLU(inplace=True),
        ) 
    def forward(self, x):
        return self.conv(x)
    
def double_conv_u1_last(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

Base =  32
Latent_Size = 64
           
class VAE_2(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()
        

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        #nn.Flatten()
        )
        
        self.z_mean = torch.nn.Linear(8*8*32*Base, Latent_Size)
        self.z_log_var = torch.nn.Linear(8*8*32*Base, Latent_Size)
        self.reshape1 = torch.nn.Linear(Latent_Size, 8*8*32*Base)
                        
        self.decoder = nn.Sequential(
                #torch.nn.Linear(64,8*8*32*Base),
                # Reshape(-1, 512, 8, 8),
                double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(32*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, 1),
                torch.nn.Sigmoid()
                )
        
    def reparameterize(self, z_mu, z_log_var):
                eps = torch.randn(z_mu.size(0), z_mu.size(1))
                eps = eps.to(device=DEVICE,dtype=torch.float) 
                z = z_mu + eps * torch.exp(z_log_var/2.).to(device=DEVICE,dtype=torch.float) 
                return z
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        #z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        #encoded = self.reparameterize(z_mean, z_log_var)  
        encoded = self.decoder(x)
        
        return encoded
    
# Input_Image_Channels = 1
# def model() -> VAE_1:
#     model = VAE_1()
#     return model
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])

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
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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

class UNet(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(UNet, self).__init__()
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
        self.outc = OutConv(32,1)
                                
        self.act = nn.Sigmoid()
        
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
         
        # return self.act(logits1)
        return logits1
        

Base =  32
Latent_Size = 64

class VAE_3(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()
        

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        nn.Flatten()
        )
        
        self.z_mean = torch.nn.Linear(8*8*32*Base, Latent_Size)
        self.z_log_var = torch.nn.Linear(8*8*32*Base, Latent_Size)
        self.reshape1 = torch.nn.Linear(Latent_Size, 8*8*32*Base)
                        
        self.decoder = nn.Sequential(
                torch.nn.Linear(64,8*8*32*Base),
                Reshape(-1, 32*Base, 8, 8),
                double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(32*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, 1),
                torch.nn.Sigmoid()
                )
        
    def reparameterize(self, z_mu, z_log_var):
                eps = torch.randn(z_mu.size(0), z_mu.size(1))
                eps = eps.to(device=DEVICE,dtype=torch.float) 
                z = z_mu + eps * torch.exp(z_log_var/2.).to(device=DEVICE,dtype=torch.float) 
                return z
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)  
        encoded = self.decoder(encoded)
        
        return encoded, z_mean, z_log_var
        
        

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):

    def __init__(self, in_chn=1, out_chn=1, BN_momentum=0.5):
        super(SegNet, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)
        
        self.act = nn.Sigmoid()

    def forward(self, x):

        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)
        return  self.act(x)
    
# Input_Image_Channels = 1
# def model() -> SegNet:
#     model = SegNet()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])