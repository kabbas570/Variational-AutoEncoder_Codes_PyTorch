import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F

from helper_data import get_dataloaders_mnist
from helper_train import train_vae_v1
from helper_utils import set_deterministic, set_all_seeds
from helper_plotting import plot_training_loss
from helper_plotting import plot_generated_images
from helper_plotting import plot_latent_space_with_labels
from helper_plotting import plot_images_sampled_from_vae

# Device
CUDA_DEVICE_NUM = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', DEVICE)

# Hyperparameters
RANDOM_SEED = 123
BATCH_SIZE = 256

set_deterministic
set_all_seeds(RANDOM_SEED)

##########################
### Dataset
##########################

train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    batch_size=BATCH_SIZE, 
    num_workers=2, 
    validation_fraction=0.)
    

# Checking the dataset
print('Training Set:\n')
for images, labels in train_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break
    
# Checking the dataset
print('\nValidation Set:')
for images, labels in valid_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break

# Checking the dataset
print('\nTesting Set:')
for images, labels in test_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break
    
#class Reshape(nn.Module):
#    def __init__(self, *args):
#        super().__init__()
#        self.shape = args
#
#    def forward(self, x):
#        return x.view(self.shape)
#
#
#class Trim(nn.Module):
#    def __init__(self, *args):
#        super().__init__()
#
#    def forward(self, x):
#        return x[:, :, :28, :28]
#
#
#class VAE(nn.Module):
#    def __init__(self):
#        super().__init__()
#        
#        self.encoder = nn.Sequential(
#                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
#                nn.LeakyReLU(0.01),
#                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
#                nn.LeakyReLU(0.01),
#                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
#                nn.LeakyReLU(0.01),
#                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
#                nn.Flatten(),
#        )    
#        
#        self.z_mean = torch.nn.Linear(3136, 2)
#        self.z_log_var = torch.nn.Linear(3136, 2)
#        
#        self.decoder = nn.Sequential(
#                torch.nn.Linear(2, 3136),
#                Reshape(-1, 64, 7, 7),
#                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
#                nn.LeakyReLU(0.01),
#                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),                
#                nn.LeakyReLU(0.01),
#                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
#                nn.LeakyReLU(0.01),
#                nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0), 
#                Trim(),  # 1x29x29 -> 1x28x28
#                nn.Sigmoid()
#                )
#
#    def encoding_fn(self, x):
#        x = self.encoder(x)
#        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
#        encoded = self.reparameterize(z_mean, z_log_var)
#        return encoded
#        
#    def reparameterize(self, z_mu, z_log_var):
#        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
#        z = z_mu + eps * torch.exp(z_log_var/2.) 
#        return z
#        
#    def forward(self, x):
#        x = self.encoder(x)
#        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
#        encoded = self.reparameterize(z_mean, z_log_var)
#        decoded = self.decoder(encoded)
#        return encoded, z_mean, z_log_var, decoded
#set_all_seeds(RANDOM_SEED)
#model = VAE()

from monai.networks.nets import VarAutoEncoder
model = VarAutoEncoder(
        spatial_dims=2,
        in_shape=(1,28,28),
        out_channels=1,
        latent_size=2,
        channels=(32,64,64,64),
        strides=(1, 2, 2,1),
    )
model_1 = model

#Input_Image_Channels = 1
#
#from torchsummary import summary
#summary(model_1, [(Input_Image_Channels, 28,28)])

### uper part is fine ###

Max_Epochs = 50
LEARNING_RATE=0.0005
Patience = 5

        #### Import All libraies used for training  #####
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
#import pandas as pd
from Early_Stopping import EarlyStopping
import matplotlib.pyplot as plt
            ### Data_Generators ########
            
            #### The first one will agument and Normalize data and used for training ###
            #### The second will not apply Data augmentaations and only prcocess the data during validation ###


   #######################################
   
print(len(train_loader)) ### this shoud be = Total_images/ batch size
print(len(train_loader))   ### same here
#print(len(test_loader))   ### same here

### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses1 = []   # losses of all training epochs
avg_valid_losses1 = []  #losses of all training epochs
avg_valid_DS1 = []  # all training epochs

### Next we have all the funcitons which will be called in the main for training ####

beta = 100  # KL beta weighting. increase for disentangled VAE


    
### 2- the main training fucntion to update the weights....
def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler,loss_fn):  ### Loader_1--> ED and Loader2-->ES
    train_losses1 = [] # loss of each batch
    valid_losses1 = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.train()
    for batch_idx, (imgs, _) in enumerate(loop):
        
        imgs = imgs.to(device=DEVICE,dtype=torch.float) 
        
        with torch.cuda.amp.autocast():
            # encoded, z_mean, z_log_var, decoded  = model1(imgs)
            # loss1 = loss_fn(z_mean, z_log_var, decoded,imgs)
             
            recon_batch, z_mean, z_log_var, _ = model(imgs)
            loss1 = loss_fn(z_mean, z_log_var, recon_batch,imgs)
             
            
            # backward
        loss = loss1
        optimizer1.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer1)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses1.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    for batch_idx, (imgs, _) in enumerate(loop_v):
        imgs = imgs.to(device=DEVICE,dtype=torch.float) 
        with torch.no_grad():
            # encoded, z_mean, z_log_var, decoded  = model1(imgs)
            # loss1 = loss_fn(z_mean, z_log_var, decoded,imgs)
            
            recon_batch, z_mean, z_log_var, _ = model(imgs)
            loss1 = loss_fn(z_mean, z_log_var, recon_batch,imgs)
            
        # backward
        loss = loss1
        loop_v.set_postfix(loss = loss.item())
        valid_losses1.append(float(loss))

    train_loss_per_epoch1 = np.average(train_losses1)
    valid_loss_per_epoch1 = np.average(valid_losses1)
    ## all epochs
    avg_train_losses1.append(train_loss_per_epoch1)
    avg_valid_losses1.append(valid_loss_per_epoch1)
    
    return train_loss_per_epoch1, valid_loss_per_epoch1


def check_Acc(loader, model1,loss_fn, device=DEVICE):
    loop = tqdm(loader)
    loss = 0
    for batch_idx, (imgs, _) in enumerate(loop):
        imgs = imgs.to(device=DEVICE,dtype=torch.float) 
        
        with torch.no_grad(): 
            # encoded, z_mean, z_log_var, decoded  = model1(imgs)
            # loss1 = loss_fn(z_mean, z_log_var, decoded,imgs)
            
            recon_batch, z_mean, z_log_var, _ = model(imgs)
            loss1 = loss_fn(z_mean, z_log_var, recon_batch,imgs)

            loss1 = float(loss1)
            loss +=loss1
    
    # print(f"Abg Erros is   : {loss1}")
    print(f"Dice_RV  : {loss/len(loader)}")
    return loss/len(loader)
    
### 6 - This is Focal Tversky Loss loss function ### 


def loss_func(z_mean, z_log_var, decoded,features):
    loss_fn = F.mse_loss
    reconstruction_term_weight = 1
    kl_div = -0.5 * torch.sum(1 + z_log_var 
                              - z_mean**2 
                              - torch.exp(z_log_var), 
                              axis=1) # sum over latent dimension

    batchsize = kl_div.size(0)
    kl_div = kl_div.mean() # average over batch dimension

    pixelwise = loss_fn(decoded, features, reduction='none')
    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
    pixelwise = pixelwise.mean() # average over batch dimension
    
    loss = reconstruction_term_weight*pixelwise + kl_div
    
    return loss
    
    
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=Patience, verbose=True)


path_to_save_Learning_Curve='/data/scratch/acw676/VAE_weights/'+'/VAE_Monai'
path_to_save_check_points='//data/scratch/acw676/VAE_weights/'+'/VAE_Monai'  ##these are weights
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
#def main():
#    model1 = model_1.to(device=DEVICE,dtype=torch.float)
#    
#            ## Fine Tunnning Part ###
##    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
##    weights_paths= "/data/home/acw676/MM/weights/a_3_three.pth.tar"
##    checkpoint = torch.load(weights_paths,map_location=DEVICE)
##    model.load_state_dict(checkpoint['state_dict'])
##    optimizer.load_state_dict(checkpoint['optimizer'])
#
#    #loss_fn1 = DiceLoss()
#    optimizer1 = optim.Adam(model1.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
#    #optimizer1 = optim.SGD(model1.parameters(),momentum=0.99,lr=LEARNING_RATE)
#    scaler = torch.cuda.amp.GradScaler()
#    loss_function = loss_func
#    for epoch in range(Max_Epochs):
#        train_loss,valid_loss=train_fn(train_loader,train_loader, model1, optimizer1,scaler,loss_function)
#        
#        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
#                     f'train_loss: {train_loss:.5f} ' +
#                     f'valid_loss: {valid_loss:.5f}')
#        
#        print(print_msg)
#
#        dice_score = check_Acc(train_loader, model1,loss_function, device=DEVICE)
#        
#        
#        avg_valid_DS1.append(dice_score)
#        
#        early_stopping(valid_loss, dice_score)
#        if early_stopping.early_stop:
#            print("Early stopping Reached at  :",epoch)
#            
#            ### save model    ######
#            checkpoint = {
#                "state_dict": model1.state_dict(),
#                "optimizer":optimizer1.state_dict(),
#            }
#            save_checkpoint(checkpoint)
#
#            break

def main():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    optimizer1 = optim.Adam(model1.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    loss_function = loss_func
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader,train_loader, model1, optimizer1,scaler,loss_function)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)

        dice_score = check_Acc(train_loader, model1,loss_function, device=DEVICE)
        
        
        avg_valid_DS1.append(dice_score)
        
        # early_stopping(valid_loss, dice_score)
        
        checkpoint = {
                "state_dict": model1.state_dict(),
                "optimizer":optimizer1.state_dict(),
            }
        save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()

### This part of the code will generate the learning curve ......

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses1)+1),avg_train_losses1, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses1)+1),avg_valid_losses1,label='Validation Loss')
plt.plot(range(1,len(avg_valid_DS1)+1),avg_valid_DS1,label='Validation DS')

# find position of lowest validation loss
minposs = avg_valid_losses1.index(min(avg_valid_losses1))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses1)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
