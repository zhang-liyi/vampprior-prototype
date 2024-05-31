import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

#=======================================================================================================================

#=======================================================================================================================
# WEIGHTS INITS
#=======================================================================================================================
def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)

#=======================================================================================================================

#=======================================================================================================================
# ACTIVATIONS
#=======================================================================================================================
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat( F.relu(x), F.relu(-x), 1 )

#=======================================================================================================================
# LAYERS
#=======================================================================================================================
class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h

#=======================================================================================================================
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, bias=True):
        super(Conv2d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
class Conv2dBN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(Conv2dBN, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

#=======================================================================================================================
class ResUnitBN(nn.Module):
    '''
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", https://arxiv.org/abs/1603.05027
    The unit used here is called the full pre-activation.
    '''
    def __init__(self, number_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ReLU(True)):
        super(ResUnitBN, self).__init__()

        self.activation = activation
        self.bn1 = nn.BatchNorm2d(number_channels)

        self.conv1 = nn.Conv2d(number_channels, number_channels, kernel_size, stride, padding, dilation)

        self.bn2 = nn.BatchNorm2d(number_channels)
        self.conv2 = nn.Conv2d(number_channels, number_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        residual = x

        h_bn_1 = self.bn1(x)
        h_act_bn_1 = self.activation(h_bn_1)
        h_1 = self.conv1(h_act_bn_1)

        h_bn_2 = self.bn2(h_1)
        h_act_bn_2 = self.activation(h_bn_2)
        h_2 = self.conv2(h_act_bn_2)

        out = h_2 + residual

        return out

# =======================================================================================================================
class ResizeConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, scale_factor=2, activation=None):
        super(ResizeConv2d, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
class ResizeConv2dBN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeConv2dBN, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        h = self.bn(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
class ResizeGatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeGatedConv2d, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = GatedConv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, activation=activation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        out = self.conv(h)

        return out

# =======================================================================================================================
class GatedConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g

# =======================================================================================================================
class GatedResUnit(nn.Module):
    def __init__(self, input_channels, activation=None):
        super(GatedResUnit, self).__init__()

        self.activation = activation
        self.conv1 = GatedConv2d(input_channels, input_channels, 3, 1, 1, 1, activation=activation)
        self.conv2 = GatedConv2d(input_channels, input_channels, 3, 1, 1, 1, activation=activation)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)

        return h2 + x

#=======================================================================================================================
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

#=======================================================================================================================
class MaskedGatedConv2d(nn.Module):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedGatedConv2d, self).__init__()

        self.h = MaskedConv2d(mask_type, *args, **kwargs)
        self.g = MaskedConv2d(mask_type, *args, **kwargs)

    def forward(self, x):
        h = self.h(x)
        g = nn.Sigmoid()(self.g(x))
        return h * g

#=======================================================================================================================
class MaskedResUnit(nn.Module):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedResUnit, self).__init__()

        self.act = nn.ReLU(True)

        self.h1 = MaskedConv2d(mask_type, *args, **kwargs)
        self.h2 = MaskedConv2d(mask_type, *args, **kwargs)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        h1 = self.bn1(x)
        h1 = self.act(h1)
        h1 = self.h1(h1)

        h2 = self.bn2(h1)
        h2 = self.act(h2)
        h2 = self.h2(h2)
        return x + h2
    

#========================================================================================================================
    
class DCGANConv(nn.Module):
    # Conv block to be used in DCGAN-style encoder
    # Each block consists of: Conv2D - BatchNorm - LeakyReLU
    def __init__(self, in_filter, out_filter):
        super().__init__()
        self.conv2d = nn.Conv2d(in_filter, out_filter, 
                                (4, 4), 
                                stride=(2, 2), 
                                padding=0)
        self.batchnorm = nn.BatchNorm2d(out_filter)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    
class DCGANConvT(nn.Module):
    # ConvTranspose block to be used in DCGAN-style decoder
    # Each block consists of: Conv2DTranspose - BatchNorm - LeakyReLU
    def __init__(self, in_filter, out_filter):
        super().__init__()
        self.conv2dtranspose = nn.ConvTranspose2d(in_filter, out_filter,
                                                               (4, 4), 
                                                               stride=(2, 2), 
                                                               padding=1)
        self.batchnorm = nn.BatchNorm2d(out_filter)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv2dtranspose(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    
class DCGANEncoder(nn.Module):
    # DCGAN-style encoder
    # It uses the previously defined DCGANConv blocks.
    def __init__(self, args, latent_dim, filter_size=64):
        super().__init__()
        self.args = args
        self.nf = filter_size
        self.dcgan_conv1 = DCGANConv(3, self.nf)
        self.dcgan_conv2 = DCGANConv(self.nf, self.nf*2)
        self.dcgan_conv3 = DCGANConv(self.nf*2, self.nf*4)
        self.conv_final = nn.Conv2d(self.nf*4, latent_dim*2, 
                                                 (4, 4), 
                                                 stride=(1, 1), 
                                                 padding=1)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = x.view((-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2]))
        x = self.dcgan_conv1(x)
        x = self.dcgan_conv2(x)
        x = self.dcgan_conv3(x)
        x = self.conv_final(x)
        x = self.flatten(x)
        return x

class DCGANDecoder(nn.Module):
    
    def __init__(self, args, latent_dim, filter_size=64, output_channel=3):
        super().__init__()
        self.nf = filter_size # filter size of the second-to-last conv transpose layer
        self.nc = output_channel # 3 for CIFAR10
        self.latent_dim = latent_dim
        
        self.conv_t1 = nn.ConvTranspose2d(latent_dim, self.nf*4, 
                                                       (4, 4), 
                                                       stride=(1, 1), 
                                                       padding=0)
        self.batchnorm = nn.BatchNorm2d(self.nf*4)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dcgan_conv_t1 = DCGANConvT(self.nf*4, self.nf*2)
        self.dcgan_conv_t2 = DCGANConvT(self.nf*2, self.nf)
        self.conv_t2 = nn.ConvTranspose2d(self.nf, self.nc, 
                                                       (4, 4), 
                                                       stride=(2, 2), 
                                                       padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = x.view((-1,self.latent_dim,1,1))
        x = self.conv_t1(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.dcgan_conv_t1(x)
        x = self.dcgan_conv_t2(x)
        x = self.conv_t2(x)
        x = self.tanh(x)
        x = (x+1)/2
        return x
