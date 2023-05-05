import torch.nn as nn
import torch


class Move_Axis(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.source,self.destination = args
    def forward(self, x):
         x = torch.moveaxis(x,(self.source),(self.destination))  ## --->  [b,c,d,h,w] --> [b,c,h,w,d] 
         return x
     
class flat(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = nn.Flatten(3,4)
    def forward(self, x):
         x = self.shape(x)
         return x 

        
class Sum_Tensor(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.axis = args
    def forward(self, x):
         x = torch.sum(x, self.axis)
         return x 
     
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)  
            


class Conv_3d(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class Conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
      
class Deconv_2d(nn.Module):
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
    
class Deconv_3d(nn.Module):
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
                  
class SeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.squeeze(x, 2)
         return x   

class UnSeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.unsqueeze(x, 2)
         return x 
   
Base = 4
class Model_1(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(Model_1, self).__init__()
        self.n_channels = n_channels
        
        self.encoder_3d = nn.Sequential( 
        Conv_3d(n_channels, Base,1,1,3),
        Conv_3d(Base, Base,(1,2,2),1,3),
        Conv_3d(Base, 2*Base,1,1,3),
        Conv_3d(2*Base, 2*Base,(1,2,2),1,3),
        Conv_3d(2*Base,4*Base,1,1,3), 
        Conv_3d(4*Base, 4*Base,(1,2,2),1,3), 
        Conv_3d(4*Base,4*Base,1,1,3), 
        Conv_3d(4*Base, Base,(1,2,2),1,3), 
        )

        
        self.encoder_2d = nn.Sequential(
    
        Conv_2d(n_channels, Base,1,1,3),
        Conv_2d(Base, Base,2,1,3),
        
        Conv_2d(Base, 2*Base,1,1,3),
        Conv_2d(2*Base, 2*Base,2,1,3),
        
        Conv_2d(2*Base,4*Base,1,1,3), 
        Conv_2d(4*Base, 4*Base,2,1,3), 
        
        Conv_2d(4*Base,4*Base,1,1,3), 
        Conv_2d(4*Base, Base,2,1,3), 
        
        Conv_2d(Base, Base,1,1,3), 
        Conv_2d(Base, Base,1,0,1), 
        )
        
        
        self.decoder_2d =  nn.Sequential(
        Deconv_2d(1,4*Base,2,2),
        Conv_2d(4*Base,4*Base,1,1,3),
        Deconv_2d(4*Base,2*Base,2,2),
        Conv_2d(2*Base,2*Base,1,1,3),
        Deconv_2d(2*Base,Base,2,2),
        Conv_2d(Base,Base,1,1,3),
        Deconv_2d(Base,Base,2,2),
        Conv_2d(Base,Base,1,1,3),
        OutConv_2d(Base,1)
        )
        
        self.decoder_3d =  nn.Sequential(
        Deconv_3d(1,4*Base,(1,2,2),(1,2,2)),
        Conv_3d(4*Base,4*Base,1,1,3),
        
        Deconv_3d(4*Base,2*Base,(1,2,2),(1,2,2)),
        Conv_3d(2*Base,2*Base,1,1,3),
      
        Deconv_3d(2*Base,Base,(1,2,2),(1,2,2)),
        Conv_3d(Base,Base,1,1,3),
        
        Deconv_3d(Base,Base,(1,2,2),(1,2,2)),
        Conv_3d(Base,Base,1,1,3),
        
        OutConv_3d(Base,1)
        )
        
        self.convert_3d_to_2d = nn.Sequential(
            
            Conv_3d(Base,Base,(2,1,1),(0,0,0),(17,1,1)),
            SeQ()

            )
        
        self.convert_2d_to_3d = nn.Sequential(
            
            UnSeQ(),
            Conv_3d(4,17,(1,1,1),(0,0,0),(1,1,1)),
            

            )

    def forward(self, x_2d,x_3d):
      emb_2d= self.encoder_2d(x_2d)
      emb_3d= self.encoder_3d(x_3d)
      
      x_c3 =  self.convert_3d_to_2d(emb_3d)
      
      print(x_c3.shape)
      
      x_c2 =  self.convert_2d_to_3d(emb_2d)
      print(x_c2.shape)
      
      #  out_2d = self.decoder_2d(x_c3)
      #  out_3d = self.decoder_3d(x_c2)
      
      #  out_2d_ = self.decoder_2d(emb_2d)
      #  out_3d_ = self.decoder_3d(emb_3d)
      
      # return out_2d,out_3d,out_2d_,out_3d_
  
Input_Image_Channels = 1
def model() -> Model_1:
    model = Model_1()
    return model
from torchsummary import summary
model = model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(Input_Image_Channels,256,256),(Input_Image_Channels,17, 256,256)]) 
