import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, in_ftr, nef, lf, device='cpu'):
        super().__init__()
        self.device=device
        self.l1 = nn.Linear(in_ftr, nef)
        self.l_mean = nn.Linear(nef, lf)
        self.l_var = nn.Linear(nef, lf)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        mean = self.l_mean(h)
        var = self.l_var(h)
        var = F.softplus(var)
        return mean, var

class VariationalDecoder(nn.Module):
    def __init__(self, lf, ndf, out_ftr, device='cpu'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(lf, ndf)
        self.l2 = nn.Linear(ndf, out_ftr)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        h = self.l2(h)
        y = torch.sigmoid(h)

        return y

class VAE(nn.Module):
    def __init__(self, in_ftr, lf, nef, ndf, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = VariationalEncoder(in_ftr, nef, lf, device=device)
        self.decoder = VariationalDecoder(lf, ndf, in_ftr, device=device)

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.latent_variable(mean, var)
        y = self.decoder(z)

        return y, z

    def latent_variable(self, mean, var):
        eps = torch.randn(mean.size(), device = self.device)
        z = mean + torch.sqrt(var) * eps

        return z

    def lower_bound(self, x):
        mean, var = self.encoder(x)
        z = self.latent_variable(mean, var)
        y = self.decoder(z)

        reconst_loss = -torch.mean(
            torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y), dim=1)
        )

        latent_loss = - 1/2 * torch.mean(
            torch.sum(1 + torch.log(var) - mean**2 - var, dim=1)
        )

        loss = reconst_loss + latent_loss

        return loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(64*64, 100, 1024, 1024, device).to(device)
    
    x = torch.randn((10, 64*64), device=device)
    criterion = model.lower_bound

    loss = criterion(x)

if __name__ == '__main__':
    main()
    