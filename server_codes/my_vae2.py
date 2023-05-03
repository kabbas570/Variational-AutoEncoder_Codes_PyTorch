import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.nn.functional as F

class double_conv01(nn.Module):
    def __init__(self, in_channels, out_channels,f_size):
        super(double_conv01, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv11(nn.Module):
    def __init__(self, in_channels, out_channels, f_size,p_size,stride=2):
        super(double_conv11, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv_u1(nn.Module):
    def __init__(self, in_channels, out_channels,f_size,st_size):
        super(double_conv_u1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            ### up-sapmling part
            nn.ConvTranspose2d(out_channels,out_channels, kernel_size=2, stride=st_size),
            nn.LeakyReLU(0.01),
        ) 
    def forward(self, x):
        return self.conv(x)
    
def double_conv_u1_last(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.LeakyReLU(0.01),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.LeakyReLU(0.01),
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
        #double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        nn.Flatten()
        )
        self.act = nn.LeakyReLU()
        self.z_mean = torch.nn.Linear(16*16*16*Base, Latent_Size)
        self.z_log_var = torch.nn.Linear(16*16*16*Base, Latent_Size)                      
        self.decoder = nn.Sequential(
                torch.nn.Linear(Latent_Size,16*16*16*Base),
                nn.LeakyReLU(),
                Reshape(-1, 16*Base, 16, 16),
                #double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(16*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, 1),
                # torch.nn.Sigmoid()
                )
        
#    def reparameterize(self, z_mu, z_log_var):
#                eps = torch.randn(z_mu.size(0), z_mu.size(1))
#                eps = eps.to(device=DEVICE,dtype=torch.float) 
#                z = z_mu + eps * torch.exp(z_log_var/2.).to(device=DEVICE,dtype=torch.float) 
#                return z
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        z_mean, z_log_var = self.act(self.z_mean(x)), self.act(self.z_log_var(x))
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
import torch.nn.functional as F

class double_conv01(nn.Module):
    def __init__(self, in_channels, out_channels,f_size):
        super(double_conv01, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv11(nn.Module):
    def __init__(self, in_channels, out_channels, f_size,p_size,stride=2):
        super(double_conv11, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv_u1(nn.Module):
    def __init__(self, in_channels, out_channels,f_size,st_size):
        super(double_conv_u1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            ### up-sapmling part
            nn.ConvTranspose2d(out_channels,out_channels, kernel_size=2, stride=st_size),
            nn.LeakyReLU(0.01),
        ) 
    def forward(self, x):
        return self.conv(x)
    
def double_conv_u1_last(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.LeakyReLU(0.01),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.LeakyReLU(0.01),
    ) 

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

Base =  32
Latent_Size = 64
           
class DN_VAE_1(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        #double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        nn.Flatten(),
        torch.nn.Linear(16*16*16*Base, Latent_Size),
        # nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
                torch.nn.Linear(Latent_Size,16*16*16*Base),
                # nn.LeakyReLU(),
                Reshape(-1, 16*Base, 16, 16),
                #double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(16*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, (3,3),padding='same'),
                torch.nn.Sigmoid()
                )
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        encoded = self.decoder(x)
        
        return encoded
    
# Input_Image_Channels = 1
# def model() -> D_VAE_1:
#     model = D_VAE_1()
#     return model
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])


import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.nn.functional as F

class double_conv01(nn.Module):
    def __init__(self, in_channels, out_channels,f_size):
        super(double_conv01, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv11(nn.Module):
    def __init__(self, in_channels, out_channels, f_size,p_size,stride=2):
        super(double_conv11, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.conv(x)

class double_conv_u1(nn.Module):
    def __init__(self, in_channels, out_channels,f_size,st_size):
        super(double_conv_u1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
            nn.LeakyReLU(0.01),
            ### up-sapmling part
            nn.ConvTranspose2d(out_channels,out_channels, kernel_size=2, stride=st_size),
            nn.LeakyReLU(0.01),
        ) 
    def forward(self, x):
        return self.conv(x)
    
def double_conv_u1_last(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.LeakyReLU(0.01),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.LeakyReLU(0.01),
    ) 

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

Base =  32
Latent_Size = 64
           
class DN_VAE_2(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        #double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        #nn.Flatten(),
        #torch.nn.Linear(16*16*16*Base, Latent_Size),
        #nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
                #torch.nn.Linear(Latent_Size,16*16*16*Base),
                #nn.LeakyReLU(),
                #Reshape(-1, 16*Base, 16, 16),
                #double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(16*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, (3,3),padding='same'),
                torch.nn.Sigmoid()
                )
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        encoded = self.decoder(x)
        
        return encoded
    
# Input_Image_Channels = 1
# def model() -> DN_VAE_1:
#     model = DN_VAE_1()
#     return model
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])

Base =  32
Latent_Size = 64
           
class DN_VAE_3(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        #nn.Flatten(),
        #torch.nn.Linear(16*16*16*Base, Latent_Size),
        #nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
                #torch.nn.Linear(Latent_Size,16*16*16*Base),
                #nn.LeakyReLU(),
                #Reshape(-1, 16*Base, 16, 16),
                double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(32*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, (3,3),padding='same'),
                torch.nn.Sigmoid()
                )
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        encoded = self.decoder(x)
        
        return encoded
        

Base =  32
Latent_Size = 64
           
class DN_VAE_4(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
        double_conv01(input_channels, Base,(3,3)),
        double_conv11(1*Base, 2*Base,(3,3),(1,1)),
        double_conv11(2*Base, 4*Base,(3,3),(1,1)),
        double_conv11(4*Base, 8*Base,(3,3),(1,1)),
        double_conv11(8*Base, 16*Base,(3,3),(1,1)),
        double_conv11(16*Base, 32*Base,(3,3),(1,1)),
        #nn.Flatten(),
        #torch.nn.Linear(16*16*16*Base, Latent_Size),
        #nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
                #torch.nn.Linear(Latent_Size,16*16*16*Base),
                #nn.LeakyReLU(),
                #Reshape(-1, 16*Base, 16, 16),
                double_conv_u1(32*Base, 32*Base,(3,3),2),
                double_conv_u1(32*Base, 16*Base,(3,3),2),
                double_conv_u1(16*Base, 8*Base,(3,3),2),
                double_conv_u1(8*Base, 4*Base,(3,3),2),
                double_conv_u1(4*Base, 2*Base,(3,3),2),
                double_conv_u1_last(2*Base, 1*Base,(3,3),2),
                nn.Conv2d(Base, 1, (3,3),padding='same'),
                torch.nn.Sigmoid()
                )
        
    def forward(self, x_in):
        x = self.encoder(x_in)
        encoded = self.decoder(x)
        
        return encoded
        

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
         
        return self.act(logits1)
        # return logits1