import torch
import torch.nn as nn
            
class Conv(nn.Module):

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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
Base = 32
class Autoencoder_2D2_7(nn.Module):
    def __init__(self, n_channels = 3, bilinear=False):
        super(Autoencoder_2D2_7, self).__init__()
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            
        Conv(n_channels, Base,2,3),
        Conv(Base, Base,1,3),
        
        Conv(Base, 2*Base,2,3),
        Conv(2*Base, 2*Base,1,3),
        
        Conv(2*Base, 4*Base,2,3),
        Conv(4*Base, 4*Base,1,3),
        
        Conv(4*Base,Base,2,3),  ## this dimenssion is 16x16x32=8,196
        
        )
        
        self.FC_Part =  nn.Sequential(
            nn.Flatten(),
            
            torch.nn.Linear(16*16*Base, 8192),
            nn.InstanceNorm1d(8192),
            nn.Sigmoid(),
            
            torch.nn.Linear(8192,8192),
            nn.InstanceNorm1d(8192),
            nn.ReLU(inplace=True),
            
            Reshape(-1,32,16,16),
            
            )
        
        self.decoder =  nn.Sequential(
        
        Deconv(32,4*Base,2,2),
        Conv(4*Base,4*Base,1,3),
      
        Deconv(4*Base,2*Base,2,2),
        Conv(2*Base,2*Base,1,3),
        
        Deconv(2*Base,Base,2,2),
        Conv(Base,Base,1,3),
        
        Deconv(Base,Base,2,2),
        OutConv(Base,3),
        )
        
    def forward(self, x_in):
      x = self.encoder(x_in)
      x = self.FC_Part(x)   ### If I add this part the performance reduces 
      encoded = self.decoder(x)
      return encoded
  
    
Input_Image_Channels = 3
def model() -> Autoencoder_2D2_7:
    model = Autoencoder_2D2_7()
    return model
from torchsummary import summary
model = model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(Input_Image_Channels, 256,256)])
