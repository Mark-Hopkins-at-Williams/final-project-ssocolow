# Inspired by https://github.com/SuchismitaSahu1993/Autoencoder-on-MNIST-in-Pytorch/blob/master/Autoencoder.py

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

# Data Preprocessing

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())
        

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Defining Parameters

num_epochs = 20
batch_size = 128
model = Autoencoder()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)

for epoch in range(num_epochs):
    for data in tqdm(dataloader):
        img, label = data

        # print(img)
        # img = Variable(img)
        # print(img)

        # forward
        output = model(img)
        loss = distance(output, img)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {loss.item()}")
    # print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), "randoNew.pt")