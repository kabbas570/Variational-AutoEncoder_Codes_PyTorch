import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
import numpy as np

def get_dataloaders_mnist(batch_size, num_workers=0,
                          validation_fraction=None,
                          train_transforms=None, test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 60000)
        train_indices = torch.arange(0, 60000 - num)
        valid_indices = torch.arange(60000 - num, 60000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader
    

train_loader, test_loader, val_loader = get_dataloaders_mnist(
    batch_size=256, 
    num_workers=2, 
    validation_fraction=0.)


Max_Epochs = 100
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
print(len(val_loader))   ### same here
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
        
        optimizer1.zero_grad()

        with torch.cuda.amp.autocast():

            encoded, z_mean, z_log_var, decoded  = model1(imgs)
            loss1 = loss_function(encoded, imgs, z_mean, z_log_var, beta)

            
            
            # backward
        loss = loss1
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
            
            encoded, z_mean, z_log_var, decoded  = model1(imgs)   
            loss1 = loss_function(encoded, imgs, z_mean, z_log_var, beta)
            
            
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


def check_Acc(loader, model1,loss_function, device=DEVICE):
    Avg_Error = 0
    loop = tqdm(loader)
    model1.eval()
    for batch_idx, (imgs, _) in enumerate(loop):
        imgs = imgs.to(device=DEVICE,dtype=torch.float) 
        with torch.no_grad():
            
            encoded, z_mean, z_log_var, decoded  = model1(imgs)
            
            loss1 = loss_function(encoded, imgs, z_mean, z_log_var, beta)

            loss1 = float(loss1)
            Avg_Error += loss1

    print(f"Abg Erros is   : {Avg_Error/len(loader)}")
    return Avg_Error/len(loader)
    
### 6 - This is Focal Tversky Loss loss function ### 

BCELoss = torch.nn.BCELoss(reduction="sum")


def loss_function(recon_x, x, mu, log_var, beta):
    bce = BCELoss(recon_x, x)
    kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld
    
    
model_1 = VAE()

epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=Patience, verbose=True)


path_to_save_Learning_Curve='/data/scratch/acw676/VAE_weights/'+'/VAE1'
path_to_save_check_points='//data/scratch/acw676/VAE_weights/'+'/VAE1'  ##these are weights
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def main():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    
            ## Fine Tunnning Part ###
#    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
#    weights_paths= "/data/home/acw676/MM/weights/a_3_three.pth.tar"
#    checkpoint = torch.load(weights_paths,map_location=DEVICE)
#    model.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])

    #loss_fn1 = DiceLoss()
    optimizer1 = optim.Adam(model1.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    #optimizer1 = optim.SGD(model1.parameters(),momentum=0.99,lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model1, optimizer1,scaler,loss_function)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)

        dice_score = check_Acc(val_loader, model1,loss_function, device=DEVICE)
        
        
        avg_valid_DS1.append(dice_score)
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            
            ### save model    ######
            checkpoint = {
                "state_dict": model1.state_dict(),
                "optimizer":optimizer1.state_dict(),
            }
            save_checkpoint(checkpoint)

            break

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