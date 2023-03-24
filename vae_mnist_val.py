import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import  os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        return x[:, :, :28, :28]


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
        )    
        
        self.z_mean = torch.nn.Linear(3136, 2)
        self.z_log_var = torch.nn.Linear(3136, 2)
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(2, 3136),
                Reshape(-1, 64, 7, 7),
                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0), 
                Trim(),  # 1x29x29 -> 1x28x28
                nn.Sigmoid()
                )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        eps = eps.to(device=DEVICE,dtype=torch.float) 
        z = z_mu + eps * torch.exp(z_log_var/2.).to(device=DEVICE,dtype=torch.float) 
        return z
        
    def forward(self, x):
       
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
       
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
    
    
# Input_Image_Channels = 1
# def model() -> VAE:
#     model = VAE()
#     return model
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 28,28)])


#model = VAE()
#model.to(DEVICE)


from monai.networks.nets import VarAutoEncoder
model = VarAutoEncoder(
        spatial_dims=2,
        in_shape=(1,28,28),
        out_channels=1,
        latent_size=2,
        channels=(32,64,64,64),
        strides=(1, 2, 2,1),
    )
# model_1 = model
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0)  

path_to_checkpoints = "/data/scratch/acw676/VAE_weights/VAE_Monai.pth.tar"
save_pre_path1 = "/data/home/acw676/VAE/imgs_my/"
checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
    

def plot_images_sampled_from_vae(model, device, latent_size, unnormalizer=None, num_images=10):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size).to(device)
        # new_images = model.decoder(rand_features)
        
        new_images = model.decode_forward(rand_features)
        #color_channels = new_images.shape[1]
        #image_height = new_images.shape[2]
        #image_width = new_images.shape[3]
        
        print(new_images.shape)
        
        for k in range(num_images):
            result = new_images[k,0,:,:]

            result = result.cuda().cpu()
            result = result.numpy().copy()
    
            plt.imsave(os.path.join(save_pre_path1,str(k)+".png"),result)



for i in range(1):
    plot_images_sampled_from_vae(model=model, device=DEVICE, latent_size=2)
