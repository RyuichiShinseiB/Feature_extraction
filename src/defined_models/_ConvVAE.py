from re import A
import torch
from torch import nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, input_feature, output_feature, n_encoder_featuer, device='cpu'):
        super().__init__()
        self.device=device
        # input size is (bs, nc, 64, 64)
        self.conv_bn_relu_layer1 = self.conv_bn_relu_3times(input_feature, n_encoder_featuer * 8)
        # input size is (bs, n_encoder_featuer * 8, 32, 32)
        self.conv_bn_relu_layer2 = self.conv_bn_relu_3times(n_encoder_featuer * 8, n_encoder_featuer * 4)
        # input size is (bs, n_encoder_featuer * 4, 16, 16)
        self.conv_bn_relu_layer3 = self.conv_bn_relu_3times(n_encoder_featuer * 4, n_encoder_featuer * 2)
        # input size is (bs, n_encoder_featuer * 2, 8, 8)
        self.conv_bn_relu_layer4 = self.conv_bn_relu_3times(n_encoder_featuer * 2, n_encoder_featuer )
        # input size is (bs, n_encoder_featuer, 4, 4)
        self.output_mean = nn.Sequential(
            nn.Conv2d(n_encoder_featuer, output_feature, 4, 1, 0),
            # nn.ReLU()
        )

        self.output_var = nn.Sequential(
            nn.Conv2d(n_encoder_featuer, output_feature, 4, 1, 0),
            # nn.ReLU()
        )

    def forward(self, input):
        x = self.conv_bn_relu_layer1(input)
        x = self.conv_bn_relu_layer2(x)
        x = self.conv_bn_relu_layer3(x)
        x = self.conv_bn_relu_layer4(x)
        mean = self.output_mean(x)
        var = self.output_var(x)
        var = F.softplus(var)

        return mean, var

    def conv_bn_relu(self, input_feature, output_feature, kernel_size, stride, padding):
        """conv_bn_relu

        Args:
            input_feature (int): input data feature
            output_feature (int): output data feature
            kernel_size (int): kernel size
            stride (int): stride
            padding (int): padding 

        Returns:
            nn.Sequential: Sequential of Conv2d, BatchNorm2d and ReLU
        
        Note:
            if you want half output size, (kernel_size, stride, pading) = (4, 2, 1)\n
            if you want same output size, (kernel_size, stride, pading) = (3, 1, 1)
        """        
        return nn.Sequential(
                    nn.Conv2d(input_feature, output_feature, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(output_feature,),
                    nn.ReLU(inplace=True)
                )

    def conv_bn_relu_3times(self, input_feature, output_feature):
        return nn.Sequential(
                    self.conv_bn_relu(input_feature, output_feature, 3, 1, 1),
                    self.conv_bn_relu(output_feature, output_feature, 3, 1, 1),
                    self.conv_bn_relu(output_feature, output_feature, 3, 1, 1),
                    nn.Conv2d(output_feature, output_feature, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(output_feature),
                    nn.ReLU(True)
                )

class VariationalDecoder(nn.Module):
    def __init__(self, input_feature, output_feature, n_decoder_feature, device='cpu'):
        super().__init__()
        self.device = device
        # input size is (bs, input_feature, 1, 1)
        self.convT_bn_selu_layer1 = self.convT_bn_selu_3times(input_feature, n_decoder_feature)
        # input size is (bs, n_decoder_feature, 2, 2)
        self.convT_bn_selu_layer2 = self.convT_bn_selu_3times(n_decoder_feature, n_decoder_feature * 2)
        # input size is (bs, n_decoder_feature * 2, 4, 4)
        self.convT_bn_selu_layer3 = self.convT_bn_selu_3times(n_decoder_feature * 2, n_decoder_feature * 4)
        # input size is (bs, n_decoder_feature * 4, 8, 8)
        self.convT_bn_selu_layer4 = self.convT_bn_selu_3times(n_decoder_feature * 4, n_decoder_feature * 8)
        # input size is (bs, n_decoder_feature * 4, 16, 16)
        self.convT_bn_selu_layer5 = self.convT_bn_selu_3times(n_decoder_feature * 8, n_decoder_feature * 8)
        # input size is (bs, n_decoder_feature * 8, 32, 32)
        self.convT_bn_selu_layer6 = nn.Sequential(
            nn.ConvTranspose2d(n_decoder_feature * 8, output_feature, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.convT_bn_selu_layer1(input)
        x = self.convT_bn_selu_layer2(x)
        x = self.convT_bn_selu_layer3(x)
        x = self.convT_bn_selu_layer4(x)
        x = self.convT_bn_selu_layer5(x)
        x = self.convT_bn_selu_layer6(x)

        return x

    def convT_bn_selu(self, input_feature, output_feature, kernel_size, stride, padding):
        """conv_bn_relu

        Args:
            input_feature (int): input data feature
            output_feature (int): output data feature
            kernel_size (int): kernel size
            stride (int): stride
            padding (int): padding 

        Returns:
            nn.Sequential: Sequential of Conv2d, BatchNorm2d and ReLU
        
        Note:
            if you want two times output size, (kernel_size, stride, pading) = (4, 2, 1)\n
            if you want same output size, (kernel_size, stride, pading) = (3, 1, 1)
        """        
        return nn.Sequential(
                    nn.ConvTranspose2d(input_feature, output_feature, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(output_feature,),
                    nn.SELU(inplace=True)
                )

    def convT_bn_selu_3times(self, input_feature, output_feature):
        return nn.Sequential(
                    self.convT_bn_selu(input_feature, output_feature, 3, 1, 1),
                    self.convT_bn_selu(output_feature, output_feature, 3, 1, 1),
                    self.convT_bn_selu(output_feature, output_feature, 4, 2, 1),
                )

class MyBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input = torch.where(torch.isnan(input), torch.zeros_like(input), input)
        input = torch.where(torch.isinf(input), torch.zeros_like(input), input)
        input = torch.where(input>1, torch.ones_like(input), input)
        target = target.float()

        return self.bce(input, target)

class VAE(nn.Module):
    def __init__(self, input_feature, output_feature, n_feature, n_base_feature, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = VariationalEncoder(input_feature, n_feature, n_base_feature)
        self.decoder = VariationalDecoder(n_feature, output_feature, n_base_feature)

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        return y, z

    def reparameterize(self, mean, var):
        eps = torch.randn(mean.size(), device=self.device)
        z = mean + torch.sqrt(var) * eps

        return z

    def lower_bound(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        # x = x.view(x.size(0), -1)
        # y = y.view(y.size(0), -1)

        # bce = x * torch.log(y) + (1 - x) * torch.log(1 - y)
        # print('bce:', bce)
        # print('bce min:', torch.min(bce, ))
        # reconst = -torch.mean(
        #     torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y),
        #                 dim=1)
        # )
        # print('reconst')
        # print(reconst,'\n')
        J_rec = MyBCELoss()
        reconst = J_rec(x, y)

        kl = - (1.0 / 2.0) * torch.mean(
            torch.sum(1.0 + torch.log(var) - mean**2 - var,
                        dim=2)
        )
        # print('kl')
        # print(kl)
        L = reconst + kl

        return L

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # encoder = VariationalEncoder(1, 10, 64, device=device).to(device=device)
    # decoder = VariationalDecoder(10, 1, 64, device=device).to(device=device)
    vae = VAE(1,1,10,64,device).to(device)
    criterion = vae.lower_bound

    x = torch.randn(5,1,64,64, device=device)
    x = torch.sigmoid(x)

    y, z= vae(x)


    # print(type(y),type(z),type(mean),type(var))
    # print('y mean:', y.mean())
    # log_y = torch.log(y)
    # log_1y = torch.log(1-y)
    # print('log(y):', log_y)
    # print('log(1-y):', log_1y)
    # print('z mean:',z.mean())
    # print('mean mean:',mean.mean(),)
    # print('var mean:',var.mean())
    loss = criterion(x)
    print('loss:',loss)
