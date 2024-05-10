# from https://github.com/pierpaolo28/Artificial-Intelligence-Projects/blob/master/Online%20Learning/ONNX/VAE.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#import onnx
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transforms)

test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transforms
)

BATCH_SIZE, N_EPOCHS, lr = 64, 10, 1e-3
INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 28*28, 256, 20                

train_iterator = DataLoader(train_dataset.data, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset.data, batch_size=BATCH_SIZE)

class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, z_dim):
            super().__init__()

            self.linear = nn.Linear(input_dim, hidden_dim)
            self.mu = nn.Linear(hidden_dim, z_dim)
            self.var = nn.Linear(hidden_dim, z_dim)

        def forward(self, x):
            hidden = F.relu(self.linear(x))
            z_mu = self.mu(hidden)
            z_var = self.var(hidden)
            return z_mu, z_var
        
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        predicted = torch.sigmoid(self.out(hidden))
        return predicted
    
    
class VAE(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x):
            z_mu, z_var = self.enc(x)
            std = torch.exp(z_var / 2)
            eps = torch.randn_like(std)
            x_sample = eps.mul(std).add_(z_mu)
            predicted = self.dec(x_sample)
            return predicted, z_mu, z_var

encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VAE(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, train_iterator, optimizer):
        model.train()
        train_loss = 0
        for i, x in enumerate(train_iterator):
            x = x.view(-1, 28 * 28)
            optimizer.zero_grad()
            x_sample, z_mu, z_var = model(x.to(device))
            recon_loss = F.binary_cross_entropy(x_sample, x.to(device), size_average=False)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            loss = recon_loss + kl_loss
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        return train_loss

def test(model, test_iterator):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, x in enumerate(test_iterator):
            x = x.view(-1, 28 * 28)
            x_sample, z_mu, z_var = model(x.to(device))
            recon_loss = F.binary_cross_entropy(x_sample, x.to(device), size_average=False)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    return test_loss

for i in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer)
    test_loss = test(model, test_iterator)
    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)
    print('Epoch ' + str(i) + ', Train Loss: ' + str(round(train_loss, 2)) + ' Test Loss: ' + str(round(test_loss, 2)))

torch.save(model.state_dict(), "trained.pt")
