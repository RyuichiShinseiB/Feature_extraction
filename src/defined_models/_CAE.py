import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import numpy as np
import matplotlib.pyplot as plt
import time


# make tensor.view() Module to use it in Sequential
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4*4*64, 128), nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 4*4*64), nn.Tanh(),
            Reshape(-1, 64, 4, 4),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.Tanh(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z

# generate and save img
def generate_and_save_images(pic, epoch):
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pic.cpu().data[i, :, :, :].permute(1, 2, 0))
        plt.axis('off')
    if epoch == 0:
        plt.savefig('./result_torch/test_sample.png')
    else:
        plt.savefig('./result_torch/image_at_epoch_{:04d}.png'.format(epoch))

# load cifar10 data in torchvision.datasets
def prepare_data(batch_size):
    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    n_sample = len(trainset)
    train_size = int(n_sample * 0.8)
    print("train data: {}, validation data: {}".format(train_size, n_sample-train_size))
    subset1_indices = list(range(0,train_size))
    subset2_indices = list(range(train_size,n_sample))
    train_dataset = Subset(trainset, subset1_indices)
    val_dataset   = Subset(trainset, subset2_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    data_loaders = {"train": train_loader, "val": val_loader}
    return data_loaders
