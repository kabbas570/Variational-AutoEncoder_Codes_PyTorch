import torch
import torch.nn as nn
            
class DoubleConv(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.Sigmoid(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=2),
            DoubleConv(out_channels, out_channels)
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


Base = 16
class Autoencoder_2D2_2(nn.Module):
    def __init__(self, n_channels = 4, bilinear=False):
        super(Autoencoder_2D2_2, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
            
        DoubleConv(n_channels, Base),
        Down(Base,2*Base),
        Down(2*Base, 4*Base),
        Down(4*Base, 8*Base),
        )
        
        self.decoder =  nn.Sequential(
        Up(8*Base, 4*Base // factor, bilinear),
        Up(4*Base, 2*Base // factor, bilinear),
        Up(2*Base, Base // factor, bilinear),
        
        OutConv(Base,4),
        )
    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded
   
#Input_Image_Channels = 4
#def model() -> Autoencoder_2D2_2:
#    model = Autoencoder_2D2_2()
#    return model
#from torchsummary import summary
#model = model()
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device=DEVICE,dtype=torch.float)
#summary(model, [(Input_Image_Channels, 256,256)])



class Down_s4(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=4),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
        

class Autoencoder_2D2_MLP(nn.Module): ## MLP based wiht 3 classes
    def __init__(self, n_channels = 3, bilinear=False):
        super(Autoencoder_2D2_MLP, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        self.encoder = nn.Sequential(
            
        DoubleConv(n_channels, Base),
        Down_s4(Base,2*Base),
        Down(2*Base, 3*Base),
        Down(3*Base, 4*Base),
        nn.Flatten(),
        torch.nn.Linear(64*16*16, 4096),
        nn.InstanceNorm1d(4096),
        torch.nn.Linear(4096,4096),
        nn.Sigmoid()
        )
        
        self.decoder =  nn.Sequential(
            
            torch.nn.Linear(4096, 4096),
            nn.InstanceNorm1d(4096),
            nn.Sigmoid(),
            torch.nn.Linear(4096, 256*256*3),
            Reshape(-1,3,256,256),
            
        )
    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded
      
Base = 16

class Autoencoder_2D2_S4(nn.Module): ## MLP based wiht 3 classes
    def __init__(self, n_channels = 3, bilinear=False):
        super(Autoencoder_2D2_S4, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        self.encoder = nn.Sequential(
            
        DoubleConv(n_channels, Base),
        Down_s4(Base,2*Base),
        Down(2*Base, 3*Base),
        Down(3*Base, 4*Base),
        #Down(4*Base, 4*Base),
        )
        
        self.decoder =  nn.Sequential(
            Up(4*Base, 3*Base // factor, bilinear),
            Up(3*Base, 3*Base // factor, bilinear),
            #Up(3*Base, 2*Base // factor, bilinear),
            Up(2*Base, 2*Base // factor, bilinear),
            Up(2*Base, Base // factor, bilinear),
            OutConv(Base,3),
        )
        
    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded