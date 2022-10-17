
import argparse

from PIL import Image
import pandas as pd
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



path = 'C://Users/jliv/Abstract_gallery/Abstract_gallery'




imlist = listdir(path)
imlist.sort()

#An image looks like the below, in each of its color channels
im = Image.open(path+"/Abstract_image_78.jpg").convert('RGB')
size=im.size

matrix = np.array(im.getdata(),dtype=np.int32).T

m1 = matrix.reshape((1,3,im.size[1],im.size[0]))

plt.imshow(m1[0][0])
plt.show()
plt.imshow(m1[0][1])
plt.show()
plt.imshow(m1[0][2])
plt.show()

plt.imshow(m1[0].T.reshape(im.size[0],im.size[1],3))
plt.show()

#The three channel image passes through convolution as so:
layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=15, stride=5, padding=5, 
 dilation=1, 
 groups=1, bias=True, padding_mode='zeros')


X_torch=torch.from_numpy(np.array(m1).astype(np.float32))
        
out = layer(X_torch)

out.shape


#With these images, load pics with numpy 3-channel images, and convert to 
#Torch Tensor. These will be our real images. Here using just 5 images.

imlist = ['Abstract_image_0.jpg',
 'Abstract_image_1.jpg',
 'Abstract_image_2.jpg',
 'Abstract_image_3.jpg',
 'Abstract_image_78.jpg']

pics = []
for pic in imlist:
    
    
    im = Image.open(path+"/"+pic).convert('RGB').resize((500,500))
    size=im.size
    
    matrix = np.array(im.getdata(),dtype=np.int32).T
    
    m1 = matrix.reshape((1,3,im.size[1],im.size[0]))

    pics.append(m1)
    
pics = np.concatenate(pics)

X_torch=torch.from_numpy(pics.astype(np.float32))
out = layer(X_torch)

out.shape

#Initializing weights for the generator an discriminator networks
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
#Discriminator predicts from 0 to 1, fake to real likelihood. 
#Will train on real images with label 1, and noise with label 0
        
ndf=64
nc=3
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf , ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf , 1, 125, 1, 0, bias=False),
                nn.Sigmoid()
                )
        

    def forward(self, img):
        
        out = self.model(img)
        
        return out

d = Discriminator()
d(X_torch).shape



"""

            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            
            
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

"""

#Generator will take random vector of latent variable size and generate 
#image noise in the size of the real images.

image_channels = 3

latent_random_noise_size = 100

latent_channels1 = 25
latent_channels2 = 10
latent_channels3 = 5
latent_channels4=3

latent_size1 = 50
latent_size2 = 500

nz=100
ngf=64
nc=3
"""
# input is Z, going into a convolution
            nn.ConvTranspose2d( 100, ngf * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5,5,0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 25 x 25
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 50 x 50
            nn.ConvTranspose2d( ngf * 2, ngf, 5, 5, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 250 x 250
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 500 x 500

"""

layers = nn.Sequential(
        nn.ConvTranspose2d( 100, 3, 10, 1, 0, bias=False),
        #nn.ConvTranspose2d( 3, 3, 4, 2, 1, bias=False),
        #nn.ConvTranspose2d( 3, 3, 4, 2, 1, bias=False),
        #nn.ConvTranspose2d( 3, 3, 5, 5, 0, bias=False),
        #nn.ConvTranspose2d( 3, 3, 5, 5, 0, bias=False),
        nn.ConvTranspose2d( 3, 3, 5, 5, 0, bias=False),
        nn.ConvTranspose2d( 3, 3, 5, 5, 0, bias=False),
        nn.ConvTranspose2d( 3, 3, 4, 2, 1, bias=False),
        
        
        )

z = torch.randn(len(X_torch), 100, 1, 1, device='cpu')
z.shape
out = layers(z)
out.shape

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        
        self.model=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, ngf, 10, 1, 0, bias=False),
            nn.BatchNorm2d(ngf ),
            nn.ReLU(True),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf, ngf, 5,5,0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*4) x 25 x 25
            nn.ConvTranspose2d(ngf, ngf, 5,5,0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 250 x 250
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 500 x 500
                )

    def forward(self, z):
        
        img = self.model(z)
        return img

g = Generator()
z = torch.randn(len(X_torch), 100, 1, 1, device='cpu')
z.shape

out=g.forward(z)
out.shape



# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

adversarial_loss = torch.nn.BCELoss()


cuda = True if torch.cuda.is_available() else False

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

torch.min((X_torch/255-.5)*2)
X_torch_adj = (X_torch/255-.5)*2
#Test Discriminator forward pass on real images
out=discriminator.forward(X_torch_adj).view(-1)

#Test loss function on tensor of ones
adversarial_loss(out,
                 torch.Tensor(np.ones(len(X_torch_adj))))

#test generator forward pass on random noise


z = torch.randn(len(X_torch), 100, 1, 1, device='cpu')

z.shape
out = generator.forward(z=z)
out.shape



###For real now


#Load all images into X_torch




# Init Networks
generator = Generator()
discriminator = Discriminator()

adversarial_loss = torch.nn.BCELoss()


cuda = True if torch.cuda.is_available() else False

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(.5,.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-5, betas=(.5,.999))



imlist = listdir(path)
imlist.sort()

batch_size = 32

batch_indices= [(i,min([i+batch_size,len(imlist)])) 
    for i in range(0,len(imlist)+1,batch_size)]

batches = len(batch_indices)

batch = batch_indices[0]
e=0
print(discriminator)

del X_torch
del out
del z
for e in range(50):
    for batch in batch_indices:
        
        f"{batch}"
        print(f"Loading Batch {batch}")
        pics = []
        for pic in imlist[batch[0]:batch[1]]:
            
            
            im = Image.open(path+"/"+pic).convert('RGB').resize((500,500))
            size=im.size
            
            matrix = np.array(im.getdata(),dtype=np.int32).T
            
            m1 = matrix.reshape((1,3,im.size[1],im.size[0]))
        
            pics.append(m1)
            
        pics = np.concatenate(pics)
        
        X_torch=((torch.from_numpy(pics.astype(np.float32))/255)-.5)*2
        
        print("Batch Loaded. Training generator.")
        
        
        valid = torch.Tensor(np.ones(len(X_torch)))
        fake = torch.Tensor(np.zeros(len(X_torch)))
    
        #Train Discriminator on real
        
        discriminator.zero_grad()
        out = discriminator(X_torch).view(-1)
        errD_real = adversarial_loss(out, valid)
        errD_real.backward()
        D_x = out.mean().item()
    
        #Train Discriminator on fake
        
        
        z = torch.randn(len(X_torch), 100, 1, 1, device='cpu')
        fake_imgs = generator(z)
        out = discriminator(fake_imgs.detach()).view(-1)
        errD_fake = adversarial_loss(out, fake)
        errD_fake.backward()
        
        D_G_z1 = out.mean().item()
        errD = errD_real + errD_fake
        optimizer_D.step()
        
        #Train Generator with Real Labels
        
        generator.zero_grad()
        out = discriminator(fake_imgs).view(-1)
        errG = adversarial_loss(out, valid)
        errG.backward()
        D_G_z2 = out.mean().item()
        optimizer_G.step()
        

        FL = round(errD_fake.item(),4)
        RL = round(errD_real.item(),4)
        OL = round(errD.item(),4)
        GL = round(errG.item(),4)
        print(f"epoch {e}; fake loss: {FL}; real loss: {RL}; overall loss: {OL}; gen loss: {GL}")

        if batch[0]%64==0:  
            z = torch.randn(1, 100, 1, 1, device='cpu')
            fake_imgs = generator(z)
            
            img = (fake_imgs[0]/2+.5)*255
            img=img.detach().numpy()
            
            
            plt.imshow(img[0])
            plt.show()
        


z = torch.randn(1, 100, 1, 1, device='cpu')
fake_imgs = generator(z)

img = (fake_imgs[0]/2+.5)*255
img=img.detach().numpy()


plt.imshow(img[2])

plt.imshow(((X_torch[2]/2+.5)*255)[2])