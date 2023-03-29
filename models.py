import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.conv(x)
    
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.conv(x)

class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()
        
        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)

class VAE_T(nn.Module):
    def __init__(self):
        super(VAE_T, self).__init__()
        
        base = 16
        
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=2, padding=1),
            Conv(base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            nn.Conv2d(4*base, 64*base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64*base, 32*base, 1)
        self.encoder_logvar = nn.Conv2d(64*base, 32*base, 1)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(32*base, 64*base, 1),
            ConvTranspose(64*base, 4*base, 8),
            Conv(4*base, 4*base, 3, padding=1),
            ConvTranspose(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            ConvTranspose(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 2*base, 3, padding=1),
            ConvTranspose(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            ConvTranspose(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, base, 3, padding=1),
            ConvTranspose(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Tanh()
        )
        
        self.c1 = nn.Conv2d(32*base, 64*base, 1)
        self.u1 = ConvTranspose(64*base, 4*base, 8)
        self.c2 =  Conv(4*base, 4*base, 3, padding=1)
        self.u2 = ConvTranspose(4*base, 4*base, 4, stride=2, padding=1)
        self.c3 = Conv(4*base, 4*base, 3, padding=1)
        self.u3  = ConvTranspose(4*base, 4*base, 4, stride=2, padding=1)
        self.c4  = Conv(4*base, 2*base, 3, padding=1)
        self.u4  = ConvTranspose(2*base, 2*base, 4, stride=2, padding=1)
        self.c5  = Conv(2*base, 2*base, 3, padding=1)
        self.u5  = ConvTranspose(2*base, 2*base, 4, stride=2, padding=1)
        self.c6  = Conv(2*base, base, 3, padding=1)
        self.u6  = ConvTranspose(base, base, 4, stride=2, padding=1)
        self.c7  = nn.Conv2d(base, 1, 3, padding=1)
        
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        a = self.c1(z)
        a = self.u1(a)
        a = self.c2(a)
        a = self.u2(a)
        a = self.c3(a)
        a = self.u3(a)
        print(a.shape)
        a = self.c4(a)
        a = self.u4(a)
        a = self.c5(a)
        a = self.u5(a)
        print(a.shape)
        a = self.c6(a)
        a = self.u6(a)
        a = self.c7(a)
        print(a.shape)
        print(self.decode(z).shape)
        
        return self.decode(z), mu, logvar

class VAE_UpSamp(nn.Module):
    def __init__(self):
        super(VAE_UpSamp, self).__init__()
        
        base = 16
        
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=2, padding=1),
            Conv(base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            nn.Conv2d(4*base, 64*base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64*base, 32*base, 1)
        self.encoder_logvar = nn.Conv2d(64*base, 32*base, 1)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(32*base, 64*base, 1),
            ConvUpsampling(64*base, 4*base, 8),
            Conv(4*base, 4*base, 3, padding=1),
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 2*base, 3, padding=1),
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, base, 3, padding=1),
            ConvUpsampling(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 3, 1, padding=1),
            nn.Tanh()
        )
        self.c1 = nn.Conv2d(32*base, 64*base, 1)
        self.u1 = ConvUpsampling(64*base, 4*base, 8)
        self.c2 =  Conv(4*base, 4*base, 3, padding=1)
        self.u2 = ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1)
        self.c3 = Conv(4*base, 4*base, 3, padding=1)
        self.u3  = ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1)
        self.c4  = Conv(4*base, 2*base, 3, padding=1)
        self.u4  = ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1)
        self.c5  = Conv(2*base, 2*base, 3, padding=1)
        self.u5  = ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1)
        self.c6  = Conv(2*base, base, 3, padding=1)
        self.u6  = ConvUpsampling(base, base, 4, stride=2, padding=1)
        self.c7  = nn.Conv2d(base, 1, 3, padding=1)
        
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        print(z.shape)
        a = self.c1(z)
        a = self.u1(a)
        a = self.c2(a)
        a = self.u2(a)
        a = self.c3(a)
        a = self.u3(a)
        print(a.shape)
        a = self.c4(a)
        a = self.u4(a)
        a = self.c5(a)
        a = self.u5(a)
        print(a.shape)
        a = self.c6(a)
        a = self.u6(a)
        a = self.c7(a)
        print(a.shape)
        print(self.decode(z).shape)
        
        return self.decode(z), mu, logvar 

Input_Image_Channels = 1
def model() -> VAE_UpSamp:
    model = VAE_UpSamp()
    return model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(Input_Image_Channels, 256,256)])
