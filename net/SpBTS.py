# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from net.BasicBlock import UNetEncoder, UNetDecoder, UNetUpSamplingBlock, UNetConvBlock, ConvNormRelu
from net.BasicBlock import LeakyReLUConv2d, ReLUINSConv2d, INSResBlock, MisINSResBlock, ReLUINSConvTranspose2d



class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=4, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.001)]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, inputs):
        h = self.main(inputs)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

class Content_Discriminator(nn.Module):
    def __init__(self, feature_maps=64, levels=4, image_size=128, c_dim=4, repeat_num=6):
        super(Content_Discriminator, self).__init__()

        curr_dim = 2**(levels) * feature_maps
        if repeat_num < levels+1:
            assert 1 > 2, 'error repeat_num in Content_Discriminator'
        layers = [nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.01)]
        for i in range(repeat_num - levels -1):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2
        # for i in range(levels):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=3, stride=1, padding=1))
        #     layers.append(nn.LeakyReLU(0.01))
        #     curr_dim = curr_dim // 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        h = self.main(inputs)
        out_cls = self.conv2(h)
        return out_cls.view(out_cls.size(0), out_cls.size(1))


class ST(nn.Module):
    def __init__(self, seg_in, seg_out, syn_in, syn_out, feature_maps=64, levels=4, nc=8, norm_type='instance',
                 use_dropout=True, wskip=False, wtrans=True):
        super(ST, self).__init__()
        self.nc = nc
        self.wtrans = wtrans
        bias = True
        enc_in = syn_in
        if not self.wtrans:
            enc_in = 1
        self.encoder = UNetEncoder(enc_in, feature_maps, levels, norm_type, use_dropout, bias=bias)
        self.decoder = UNetDecoder(seg_out, feature_maps, levels, norm_type, bias=bias, wskip=wskip)

        # self.score_last2 = nn.Conv2d(feature_maps*2, seg_out, kernel_size=1, bias=bias)
        # self.score_last3 = nn.Conv2d(feature_maps*4, seg_out, kernel_size=1, bias=bias)

        if self.wtrans:
            self.syn_enc_attr = E_attr(input_dim=syn_in, output_nc=self.nc)
            self.syn_decoder = T_Decoder(output_dim=syn_out, nz=self.nc, levels=levels, feature_maps=feature_maps)



    def forward(self, *inputs):
        if len(inputs) == 4:
            return self.forward_two(inputs[0], inputs[1], inputs[2], inputs[3])
        elif len(inputs) == 2:
            return self.forward_one(inputs[0], inputs[1])
        elif len(inputs) == 3:
            return self.forward_three(inputs[0], inputs[1], inputs[2])
        else:
            assert 1 > 2, 'error for input'

    def forward_two(self, input_a_x, dmcd_x, input_b_y, dmcd_y):  # input_a_x -> image a from domain x; dmcd -> domain code
        # #
        # encoder_outputs_a, content_a = self.encoder(input_a_x)
        # encoder_outputs_b, content_b = self.encoder(input_b_y)

        # 输入处理
        dmcd_x = dmcd_x.view(dmcd_x.size(0), dmcd_x.size(1), 1, 1)
        dmcd_x = dmcd_x.repeat(1, 1, input_a_x.size(2), input_a_x.size(3))
        syn_a_x = torch.cat([input_a_x, dmcd_x], dim=1)

        dmcd_y = dmcd_y.view(dmcd_y.size(0), dmcd_y.size(1), 1, 1)
        dmcd_y = dmcd_y.repeat(1, 1, input_b_y.size(2), input_b_y.size(3))
        syn_b_y = torch.cat([input_b_y, dmcd_y], dim=1)


        #
        if not self.wtrans:
            encoder_outputs_a, content_a = self.encoder(input_a_x)
            encoder_outputs_b, content_b = self.encoder(input_b_y)
        else:
            encoder_outputs_a, content_a = self.encoder(syn_a_x)
            encoder_outputs_b, content_b = self.encoder(syn_b_y)

        d_out_a, seg_a = self.decoder(content_a, encoder_outputs_a)
        d_out_b, seg_b = self.decoder(content_b, encoder_outputs_b)

        # last two and three output
        dout_a = []
        dout_a.append(seg_a)
        # dout_a.append(self.score_last2(d_out_a[-2]))
        # dout_a.append(self.score_last3(d_out_a[-3]))

        dout_b = []
        dout_b.append(seg_b)
        # dout_b.append(self.score_last2(d_out_b[-2]))
        # dout_b.append(self.score_last3(d_out_b[-3]))

        if not self.wtrans:
            return seg_a, seg_b, None, None, None, None, content_a, content_b


        #
        # syn attribute encoder
        style_x = self.syn_enc_attr(syn_a_x)
        style_y = self.syn_enc_attr(syn_b_y)



        # syn decoder
        syn_a_y = self.syn_decoder(content_a, style_y)
        syn_b_x = self.syn_decoder(content_b, style_x)

        # reconstruct image
        rec_a_x = self.syn_decoder(content_a, style_x)
        rec_b_y = self.syn_decoder(content_b, style_y)

        return dout_a, dout_b, syn_a_y, syn_b_x, rec_a_x, rec_b_y, content_a, content_b, \
               style_x, style_y


    def forward_one(self, input, dmcd):  # input_a_x -> image a from domain x; dmcd -> domain code

        dmcd = dmcd.view(dmcd.size(0), dmcd.size(1), 1, 1)
        dmcd = dmcd.repeat(1, 1, input.size(2), input.size(3))
        syn_input = torch.cat([input, dmcd], dim=1)
        if not self.wtrans:
            encoder_outputs, content = self.encoder(input)
        else:
            encoder_outputs, content = self.encoder(syn_input)
        _, seg_map = self.decoder(content, encoder_outputs)

        if not self.wtrans:
            return seg_map, None

        style = self.syn_enc_attr(syn_input)

        # reconstruct image
        rec = self.syn_decoder(content, style)

        return seg_map, rec
        # return seg_map
    def forward_three(self, input, dmcd, sty):  # input_a_x -> image a from domain x; dmcd -> domain code

        dmcd = dmcd.view(dmcd.size(0), dmcd.size(1), 1, 1)
        dmcd = dmcd.repeat(1, 1, input.size(2), input.size(3))
        syn_input = torch.cat([input, dmcd], dim=1)

        encoder_outputs, content = self.encoder(syn_input)

        style = self.syn_enc_attr(syn_input)

        # reconstruct image
        if sty == None:
            syn_image = self.syn_decoder(content, style)
        else:
            syn_image = self.syn_decoder(content, sty)

        return syn_image, style

# ****************************************************************************************
# ------------------ Translation   E_content、 E_attr、 E_G -------------------------------
# ****************************************************************************************

class E_attr(nn.Module):
    def __init__(self, input_dim, output_nc=16):
        super(E_attr, self).__init__()
        dim = 64
        self.model = nn.Sequential(  # H and W = 216
            nn.ReflectionPad2d(3),  # H and W = 222
            nn.Conv2d(input_dim, dim, 7, 1),  # H and W = 216
            # nn.Conv2d(1, dim, 7, 1),  # H and W = 216
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  # H and W = 218
            nn.Conv2d(dim, dim*2, 4, 2),  # H and W = 108   (218+1-3)/2
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  # H and W = 110
            nn.Conv2d(dim*2, dim*4, 4, 2),  # H and W = 54
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  # H and W = 56
            nn.Conv2d(dim*4, dim*4, 4, 2),  # H and W = 27
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  # H and W = 29
            nn.Conv2d(dim*4, dim*4, 4, 2),  # H and W = 13
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # shape -> B , C , 1
            nn.Conv2d(dim*4, output_nc, 1, 1, 0))
        return

    def forward(self, x):
        x = self.model(x)
        output = x.view(x.size(0), -1)  # B C H W
        return output

class T_Decoder(nn.Module):
    def __init__(self, output_dim=1, nz=16, levels=4, feature_maps=32):
        # nz -> style code vector
        super(T_Decoder, self).__init__()
        self.nz = nz
        self.out_channels = output_dim
        self.feature_maps = feature_maps
        self.levels = levels
        in_chnnal = feature_maps * (2**levels)
        tch_add = in_chnnal
        tch = in_chnnal
        self.tch_add = tch_add


        # dec0 = []
        # dec0 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        # tch = tch // 2
        # dec0 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        # self.dec0 = nn.Sequential(*dec0)

        # dec5 = []
        # dec5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        # tch = tch//2
        # dec5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        # tch = tch//2
        # dec5 += [nn.ConvTranspose2d(tch, output_dim, kernel_size=1, stride=1, padding=0)]
        # dec5 += [nn.Tanh()]
        # self.dec5 = nn.Sequential(*dec5)

        self.dec5 = nn.Sequential()

        for i in range(levels):
            upconv = UNetUpSamplingBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                         deconv=True,
                                         bias=True)
            self.dec5.add_module('upconv%d' % (i + 1), upconv)

            conv_block = UNetConvBlock(2 ** (levels - i -1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                       norm_type='instance', bias=True)
            self.dec5.add_module('convblock%d' % (i + 1), conv_block)
        self.score = nn.Conv2d(feature_maps, output_dim, kernel_size=1, bias=True)


        self.dec1 = MisINSResBlock(tch, tch_add)
        self.dec2 = MisINSResBlock(tch, tch_add)
        self.dec3 = MisINSResBlock(tch, tch_add)
        self.dec4 = MisINSResBlock(tch, tch_add)

        self.mlp = nn.Sequential(
            nn.Linear(self.nz, tch_add),
            nn.ReLU(inplace=True),
            nn.Linear(tch_add, tch_add),
            nn.ReLU(inplace=True),
            nn.Linear(tch_add, tch_add*4))
        return

    def forward(self, x, z):  # x -> content  z -> attribute
        z = self.mlp(z)  # z.shape() -> (batch_size, 256*4)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.dec1(x, z1)
        out2 = self.dec2(out1, z2)
        out3 = self.dec3(out2, z3)
        out4 = self.dec4(out3, z4)

        # encoder_outputs.reverse()
        outputs = out4
        for i in range(self.levels):
            # outputs = getattr(self.dec5, 'upconv%d' % (i + 1))(encoder_outputs[i], outputs)
            outputs = getattr(self.dec5, 'upconv%d' % (i + 1))(outputs)
            outputs = getattr(self.dec5, 'convblock%d' % (i + 1))(outputs)
        # encoder_outputs.reverse()

        return self.score(outputs)
