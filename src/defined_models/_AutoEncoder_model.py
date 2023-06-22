import torch
import torchvision
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_feature, output_feature, n_encoder_feat):
        super().__init__()
        # input size is (bs, nc, 64, 64)
        self.conv_bn_relu_layer1 = self.conv_bn_relu_pool(input_feature, n_encoder_feat * 8, 3, 1, 1)
        # input size is (bs, n_encoder_feat * 8, 32, 32)
        self.conv_bn_relu_layer2 = self.conv_bn_relu_pool(n_encoder_feat * 8, n_encoder_feat * 4, 3, 1, 1)
        # input size is (bs, n_encoder_feat * 4, 16, 16)
        self.conv_bn_relu_layer3 = self.conv_bn_relu_pool(n_encoder_feat * 4, n_encoder_feat * 2, 3, 1, 1)
        # input size is (bs, n_encoder_feat * 2, 8, 8)
        self.conv_bn_relu_layer4 = self.conv_bn_relu_pool(n_encoder_feat * 2, n_encoder_feat , 3, 1, 1)
        # input size is (bs, n_encoder_feat, 4, 4)
        self.conv_bn_relu_layer5 = nn.Sequential(
            nn.Conv2d(n_encoder_feat, output_feature, 4, 1, 0),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.conv_bn_relu_layer1(input)
        x = self.conv_bn_relu_layer2(x)
        x = self.conv_bn_relu_layer3(x)
        x = self.conv_bn_relu_layer4(x)
        x = self.conv_bn_relu_layer5(x)

        return x

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

    def conv_bn_relu_pool(self, input_feature, output_feature, kernel_size, stride, padding):
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
            if you want quarter output size, (kernel_size, stride, pading) = (4, 2, 1)\n
            if you want half output size, (kernel_size, stride, pading) = (3, 1, 1)
        """        
        return nn.Sequential(
                    nn.Conv2d(input_feature, output_feature, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(output_feature,),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )

class Decoder(nn.Module):
    def __init__(self, input_feature, output_feature, n_decoder_feature):
        super().__init__()
        # input size is (bs, input_feature, 1, 1)
        self.convT_bn_selu_layer1 = self.convT_bn_selu(input_feature, n_decoder_feature, 4, 2, 1)
        # input size is (bs, n_decoder_feature, 2, 2)
        self.convT_bn_selu_layer2 = self.convT_bn_selu(n_decoder_feature, n_decoder_feature * 2, 4, 2, 1)
        # input size is (bs, n_decoder_feature * 2, 4, 4)
        self.convT_bn_selu_layer3 = self.convT_bn_selu(n_decoder_feature * 2, n_decoder_feature * 4, 4, 2, 1)
        # input size is (bs, n_decoder_feature * 4, 8, 8)
        self.convT_bn_selu_layer4 = self.convT_bn_selu(n_decoder_feature * 4, n_decoder_feature * 8, 4, 2, 1)
        # input size is (bs, n_decoder_feature * 4, 16, 16)
        self.convT_bn_selu_layer5 = self.convT_bn_selu(n_decoder_feature * 8, n_decoder_feature * 8, 4, 2, 1)
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
        
        Attribute:
            if you want two times output size, (kernel_size, stride, pading) = (4, 2, 1)\n
            if you want same output size, (kernel_size, stride, pading) = (3, 1, 1)
        """        
        return nn.Sequential(
                    nn.ConvTranspose2d(input_feature, output_feature, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(output_feature,),
                    nn.SELU(inplace=True)
                )
        
class CAE(nn.Module):
    def __init__(self, input_feature, output_feature, n_feature, n_base_feature, ):
        super().__init__()
        self.encoder = Encoder(input_feature, n_feature, n_base_feature)
        self.decoder = Decoder(n_feature, output_feature, n_base_feature)
    
    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    encoder = Encoder(1, 10, 64)
    decoder = Decoder(10, 1, 64)

    X = torch.randn(1, 1, 64, 64)
    out = encoder(X)

    print(out.shape)

    y = decoder(out)

    print(y.shape)

    convT = nn.ConvTranspose2d(1, 1, 3, 1, 1)

    convT_out = convT(X)
    print(convT_out.shape)
