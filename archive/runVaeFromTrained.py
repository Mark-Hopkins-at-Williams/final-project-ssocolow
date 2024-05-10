# using downloaded data
import torch
import torch.nn as nn
import torch.nn.functional as F

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

BATCH_SIZE, N_EPOCHS, lr = 64, 10, 1e-3
INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 28*28, 256, 20

encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)


pretrained = VAE(encoder, decoder)
pretrained.load_state_dict(torch.load('trained.pt'))
pretrained.eval()

pretrained.enc.forward()