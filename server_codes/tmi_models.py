import torch
import torch.nn as nn
            
class Conv(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels,stride,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)

class Deconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,stride,kernel_size):
        super().__init__()
        
        self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        x = self.up(x1)
        return  x
        
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :256, :256]

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
              
# Input_Image_Channels = 3
# def model() -> Autoencoder_2D2_2:
#     model = Autoencoder_2D2_2()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])

      
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
        
#Base = 2
#class Autoencoder_2D2_7(nn.Module):
#    def __init__(self, n_channels = 1, bilinear=False):
#        super(Autoencoder_2D2_7, self).__init__()
#        self.n_channels = n_channels
#        
#        self.encoder = nn.Sequential(
#            
#        Conv(n_channels, Base,2,3),
#        Conv(Base, Base,1,3),
#        
#        Conv(Base, 2*Base,2,3),
#        Conv(2*Base, 2*Base,1,3),
#        
#        Conv(2*Base, 4*Base,2,3),
#        Conv(4*Base, 4*Base,1,3),
#        
#        Conv(4*Base,4*Base,2,3),  ## this dimenssion is 16x16x32=8,196
#        
#        Conv(4*Base,Base,1,3),  ## this dimenssion is 16x16x32=8,196
#        
#        )
#        
#        self.FC_Part =  nn.Sequential(
#            nn.Flatten(),
#            
#            torch.nn.Linear(16*16*16, 512),
#            nn.InstanceNorm1d(512),
#            nn.ReLU(),
#            nn.Dropout(0.3),
#            
#            torch.nn.Linear(512,4096),
#            nn.InstanceNorm1d(4096),
#            nn.ReLU(inplace=True),
#            nn.Dropout(0.3),
#            
#            Reshape(-1,16,16,16),
#            
#            )
#            
#        
#        self.decoder =  nn.Sequential(
#        
#        Deconv(Base,4*Base,2,2),
#        Conv(4*Base,4*Base,1,3),
#      
#        Deconv(4*Base,2*Base,2,2),
#        Conv(2*Base,2*Base,1,3),
#        
#        Deconv(2*Base,Base,2,2),
#        Conv(Base,Base,1,3),
#        
#        Deconv(Base,Base,2,2),
#        OutConv(Base,1),
#        )
#        
#    def forward(self, x_in):
#      x = self.encoder(x_in)
#      #x = self.FC_Part(x)
#      encoded = self.decoder(x)
#      return encoded
      
      

class UnSeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.unsqueeze(x, 4)
         return x  
     
class SeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.squeeze(x, 4)
         return x  
        
Base = 32
class Autoencoder_2D2_8(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(Autoencoder_2D2_8, self).__init__()
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            
        Conv(n_channels, Base,2,3),
        Conv(Base, Base,1,3),
        
        Conv(Base, 2*Base,2,3),
        Conv(2*Base, 2*Base,1,3),
        
        Conv(2*Base, 4*Base,2,3),
        Conv(4*Base, 4*Base,1,3),
        
        Conv(4*Base,4*Base,2,3),  ## this dimenssion is 16x16x32=8,196
        Conv(4*Base,4*Base,1,3),  ## this dimenssion is 16x16x32=8,196
        
        UnSeQ(),
        torch.nn.Linear(1,17,bias=False),
        nn.ReLU(inplace=True)
        )

        self.decoder =  nn.Sequential(
            
        torch.nn.Linear(17,1,bias=False),
        nn.ReLU(inplace=True),
        SeQ(),
        
        Deconv(4*Base,4*Base,2,2),
        Conv(4*Base,4*Base,1,3),
      
        Deconv(4*Base,2*Base,2,2),
        Conv(2*Base,2*Base,1,3),
        
        Deconv(2*Base,Base,2,2),
        Conv(Base,Base,1,3),
        
        Deconv(Base,Base,2,2),
        Conv(Base,Base,1,3),
        
        OutConv(Base,1),
         )
        
    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded


# Input_Image_Channels = 1
# def model() -> Autoencoder_2D2_8:
#     model = Autoencoder_2D2_8()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])