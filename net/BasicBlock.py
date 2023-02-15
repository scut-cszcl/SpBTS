import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# ****************************************************************************************
# ----------------------------------- Unet   Basic blocks  ---------------------------------
# ****************************************************************************************
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True,
                 use_last_block=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        self.use_last_block = use_last_block
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            conv_block = UNetConvBlock(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)

            in_features = out_features
        if use_last_block:
            self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps)


    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)

            if i == self.levels - 1 and self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)
        if self.use_last_block:
            outputs = self.center_conv(outputs)
        return encoder_outputs, outputs

class UNetDecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True, wskip=True):
        super(UNetDecoder, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.wskip = wskip
        self.features = nn.Sequential()

        for i in range(levels):
            upconv = UNetUpSamplingBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)
            if self.wskip:
                conv_block = UNetConvBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps,
                                           norm_type=norm_type, bias=bias)
            else:
                conv_block = UNetConvBlock(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        decoder_outputs = []
        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            if self.wskip:
                outputs = getattr(self.features, 'upconv%d' % (i+1))(encoder_outputs[i], outputs)
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            decoder_outputs.append(outputs)
        encoder_outputs.reverse()
        return decoder_outputs, self.score(outputs)

class UNetDecoder_edecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True, wskip=True):
        super(UNetDecoder_edecoder, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.wskip = wskip
        self.features = nn.Sequential()

        for i in range(levels):
            upconv = UNetUpSamplingBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)
            if self.wskip:
                conv_block = UNetConvBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps,
                                           norm_type=norm_type, bias=bias)
            else:
                conv_block = UNetConvBlock(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)
        self.block = nn.Sequential(
            UNetConvBlock(2 ** (levels) * feature_maps, 2 ** (levels) * feature_maps,
                          norm_type='instance', bias=True),
            UNetConvBlock(2 ** (levels) * feature_maps, 2 ** (levels) * feature_maps,
                          norm_type='instance', bias=True)
        )

    def forward(self, inputs, encoder_outputs):
        decoder_outputs = []
        encoder_outputs.reverse()

        # outputs = inputs
        outputs = self.block(inputs)
        for i in range(self.levels):
            if self.wskip:
                outputs = getattr(self.features, 'upconv%d' % (i+1))(encoder_outputs[i], outputs)
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            decoder_outputs.append(outputs)
        encoder_outputs.reverse()
        return decoder_outputs, self.score(outputs)


class ST_D(nn.Module):
    def __init__(self, out_channels, input_size, feature_maps=64,
                 levels=4, norm_type='instance', bias=True, wskip=True, share_l=0):
        super(ST_D, self).__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.share_l = share_l
        self.wskip = wskip
        self.features = nn.Sequential()

        for i in range(share_l, levels):
            upconv = UNetUpSamplingBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)
            if self.wskip:
                conv_block = UNetConvBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps,
                                           norm_type=norm_type, bias=bias)
            else:
                conv_block = UNetConvBlock(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)
        # self.prefuse = UNetConvBlock(feature_maps + input_size, feature_maps//2,
        #                                    norm_type=norm_type, bias=bias)
        # self.fuse = nn.Sequential(nn.Conv2d(feature_maps//2, 1, kernel_size=3, padding=1),
        #                                nn.Sigmoid())
        # self.conv = ConvNormRelu(feature_maps, feature_maps,norm_type=norm_type, bias=bias)
        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)

    def forward(self, inputs, encoder_outputs, images):
        decoder_outputs = []
        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.share_l, self.levels):
            if self.wskip:
                outputs = getattr(self.features, 'upconv%d' % (i+1))(encoder_outputs[i], outputs)
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            decoder_outputs.append(outputs)
        encoder_outputs.reverse()
        # outputs2 = self.prefuse(torch.cat([outputs, images], dim=1))
        # active = self.fuse(outputs2)
        # outputs = self.conv(outputs)
        # return decoder_outputs, self.score(torch.mul(outputs, active))
        return decoder_outputs, self.score(outputs)

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, feature_maps=64, levels=4, norm_type='instance',
                 use_dropout=True, bias=True, wskip=True):
        super(ContentEncoder, self).__init__()
        self.cencoder = UNetEncoder(in_channels, feature_maps, levels, norm_type, use_dropout, bias=bias)
        self.cdecoder = UNetDecoder(out_channels, feature_maps, levels, norm_type, bias=bias, wskip=wskip)
        self.roundblock = RoundGradient()

    def forward(self, inputs):
        encoder_outputs, features = self.cencoder(inputs)
        _, content = self.cdecoder(features, encoder_outputs)
        return content, self.roundblock(content)

# class STDecoder(nn.Module):
#     def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
#         super(STDecoder, self).__init__()
#
#         self.out_channels = out_channels
#         self.feature_maps = feature_maps
#         self.levels = levels
#         self.features = nn.Sequential()
#
#         for i in range(levels):
#             upconv = UNetUpSamplingBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
#                                          deconv=True,
#                                          bias=bias)
#             self.features.add_module('upconv%d' % (i + 1), upconv)
#
#             conv_block = UNetConvBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
#                                        norm_type=norm_type, bias=bias)
#             self.features.add_module('convblock%d' % (i + 1), conv_block)
#
#         self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)
#         self.active = nn.Tanh()
#
#     def forward(self, inputs, encoder_outputs):
#         decoder_outputs = []
#         encoder_outputs.reverse()
#
#         outputs = inputs
#         for i in range(self.levels):
#             outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[i], outputs)
#             outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
#             decoder_outputs.append(outputs)
#         encoder_outputs.reverse()
#
#         return decoder_outputs, self.active(self.score(outputs))

class UNetUpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True):
        super(UNetConvBlock, self).__init__()

        self.conv1 = ConvNormRelu(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)
        self.conv2 = ConvNormRelu(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(ConvNormRelu, self).__init__()
        norm = nn.BatchNorm2d if norm_type == 'batch' else nn.InstanceNorm2d
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0

        self.unit = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  norm(out_channels),
                                  nn.LeakyReLU(0.01))

    def forward(self, inputs):
        return self.unit(inputs)

class Round_Gradient(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        rg = g.clone()
        return rg


class RoundGradient(nn.Module):
    def __init__(self):
        super(RoundGradient, self).__init__()
        self.s = nn.Softmax(dim=1)
    def forward(self, input):
        output = self.s(input)
        output = Round_Gradient.apply(output)
        # return output.round()
        return output

# ****************************************************************************************
# ----------------------------------- TS   Basic blocks  ---------------------------------
# ****************************************************************************************
def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(0.01)]
        # model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        #elif == 'Group'
    def forward(self, x):
        return self.model(x)

class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class MisINSResBlock(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.LeakyReLU(0.01),
            # nn.LeakyReLU(0.01, inplace=True),
            self.conv1x1(dim + dim_extra, dim),
            nn.LeakyReLU(0.01))
            # nn.LeakyReLU(0.01, inplace=True))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.LeakyReLU(0.01),
            # nn.LeakyReLU(0.01, inplace=True),
            self.conv1x1(dim + dim_extra, dim),
            nn.LeakyReLU(0.01))
            # nn.LeakyReLU(0.01, inplace=True))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)
    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out

class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                           'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
          # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                                *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u
    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))
    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module