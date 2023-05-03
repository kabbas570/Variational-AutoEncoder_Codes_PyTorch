import torch
import torch.nn as nn
            
class Conv(nn.Module):
    """(convolution => [BN] =>) * 2"""

    def __init__(self, in_channels, out_channels,stride,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=1,stride=stride),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)

class Deconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,stride,kernel_size):
        super().__init__()
        
        self.up = nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.InstanceNorm3d(out_channels),
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
                  
class Move_Axis1(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.moveaxis(x,2,4)  ## --->  [b,c,d,h,w] --> [b,c,h,w,d] 
         return x

class Move_Axis2(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.moveaxis(x,4,2)  ## --->  [b,c,d,h,w] --> [b,c,h,w,d] 
         return x

class SeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.squeeze(x, 4)
         return x   

class UnSeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.unsqueeze(x, 4)
         return x 
             
#Base = 32
#class Autoencoder_3D3_8(nn.Module):
#    def __init__(self, n_channels = 1, bilinear=False):
#        super(Autoencoder_3D3_8, self).__init__()
#        self.n_channels = n_channels
#        
#        self.encoder = nn.Sequential(
#            
#        Conv(n_channels, Base,(1,2,2),3),
#        Conv(Base, Base,1,3),
#        
#        Conv(Base, 2*Base,(1,2,2),3),
#        
#        Conv(2*Base, 4*Base,(1,2,2),3), 
#       
#         
#        Conv(4*Base,Base,1,3), 
#        
#        Move_Axis1(),
#        torch.nn.Linear(17,1,bias=False),
#        nn.ReLU(inplace=True),
#        SeQ()
#        
#        )
#        
#        self.decoder =  nn.Sequential(
#            
#        UnSeQ(),
#        torch.nn.Linear(1,17,bias=False),
#        nn.ReLU(inplace=True),
#        Move_Axis2(),
#        
#        Deconv(Base,4*Base,(1,2,2),(1,2,2)),
#        
#        Conv(4*Base,4*Base,1,3),
#      
#        Deconv(4*Base,2*Base,(1,2,2),(1,2,2)),
#        Conv(2*Base,2*Base,1,3),
#        
#        Deconv(2*Base,Base,(1,2,2),(1,2,2)),
#        Conv(Base,Base,1,3),
#        
#        Deconv(Base,Base,(1,2,2),(1,2,2)),
#        OutConv(Base,1)
#        )
#
#    def forward(self, x_in):
#      x = self.encoder(x_in)
#      encoded = self.decoder(x)
#      return encoded
     
# Input_Image_Channels = 1
# def model() -> Autoencoder_3D3_8:
#     model = Autoencoder_3D3_8()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,17, 256,256)])


Base = 64
class Autoencoder_3D3_8(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(Autoencoder_3D3_8, self).__init__()
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            
        Conv(n_channels, Base,1,3),
        Conv(Base, Base,(1,2,2),3),
        
        Conv(Base, 2*Base,1,3),
        Conv(2*Base, 2*Base,(1,2,2),3),
        
        Conv(2*Base,4*Base,1,3), 
        Conv(4*Base, 4*Base,(1,2,2),3), 
        
        Conv(4*Base,4*Base,1,3), 
        Conv(4*Base, Base,(1,2,2),3), 
         
        Move_Axis1(),
        torch.nn.Linear(17,1,bias=False),
        SeQ(),
        #nn.InstanceNorm3d(16),
        nn.ReLU(inplace=True),
        )
        
        self.decoder =  nn.Sequential(
            
        UnSeQ(),
        torch.nn.Linear(1,17,bias=False),
        Move_Axis2(),
        #nn.InstanceNorm3d(16),
        nn.ReLU(inplace=True),
        
        Deconv(Base,4*Base,(1,2,2),(1,2,2)),
        Conv(4*Base,4*Base,1,3),
        
        Deconv(4*Base,2*Base,(1,2,2),(1,2,2)),
        Conv(2*Base,2*Base,1,3),
      
        Deconv(2*Base,Base,(1,2,2),(1,2,2)),
        Conv(Base,Base,1,3),
        
        Deconv(Base,Base,(1,2,2),(1,2,2)),
        Conv(Base,Base,1,3),
        
        OutConv(Base,1)
        )

    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded
  
# Input_Image_Channels = 1
# def model() -> Autoencoder_3D3_8:
#     model = Autoencoder_3D3_8()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,17, 256,256)])