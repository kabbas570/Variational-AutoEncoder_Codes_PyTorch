import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
Latent_Size = 128
           
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
        self.decoder = nn.Sequential(
                torch.nn.Linear(Latent_Size,8*8*32*Base),
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
    
# Input_Image_Channels = 1
# def model() -> VAE_1:
#     model = VAE_1()
#     return model
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])