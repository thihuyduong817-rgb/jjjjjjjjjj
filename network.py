import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from functools import partial

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


# =================================================================================== #
# =================== 新增：ASM_Net (Angular Spectrum Method Network) ================ #
# =================================================================================== #

class LearnableInversePropagator(nn.Module):
    """
    这是ASM_Net的核心模块，一个在频域中操作的可学习网络。
    它接收一个2通道的输入（实部和虚部），并输出一个2通道的修正后的频谱。
    其目标是学习如何恢复相位并抵消衍射效应。
    此版本使用一个对称的U-Net结构来保证输入输出的空间维度一致。
    """

    def __init__(self, in_channels=2, out_channels=2, feature_channels=32):
        super(LearnableInversePropagator, self).__init__()
        # Symmetrical U-Net for frequency domain processing
        self.encoder1 = conv_block(in_channels, feature_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = conv_block(feature_channels, feature_channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = conv_block(feature_channels * 2, feature_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(feature_channels * 4, feature_channels * 2, 2, stride=2)
        self.decoder2 = conv_block(feature_channels * 4, feature_channels * 2)  # Takes concatenated input

        self.upconv1 = nn.ConvTranspose2d(feature_channels * 2, feature_channels, 2, stride=2)
        self.decoder1 = conv_block(feature_channels * 2, feature_channels)  # Takes concatenated input

        self.final_conv = nn.Conv2d(feature_channels, out_channels, 1)

    def forward(self, x):
        # Downsampling path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottle = self.bottleneck(self.pool2(enc2))

        # Upsampling path with skip connections
        up2 = self.upconv2(bottle)
        dec2_in = torch.cat([up2, enc2], dim=1)
        dec2 = self.decoder2(dec2_in)

        up1 = self.upconv1(dec2)
        dec1_in = torch.cat([up1, enc1], dim=1)
        dec1 = self.decoder1(dec1_in)

        return self.final_conv(dec1)


class ASM_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(ASM_Net, self).__init__()

        # 模块二：可学习的逆向传播模块
        self.inverse_propagator = LearnableInversePropagator()

        # 模块四：空间域精炼模块
        self.spatial_refiner = nn.Sequential(
            conv_block(ch_in=img_ch, ch_out=64),
            conv_block(ch_in=64, ch_out=32),
            nn.Conv2d(32, output_ch, kernel_size=1)
        )

    def forward(self, x):
        # 输入x是衍射图像的强度 (B, 1, H, W)
        # 我们假设输入强度是振幅的平方，先开方得到振幅
        amplitude = torch.sqrt(torch.clamp(x, min=1e-8))

        # 由于没有相位信息，我们假设初始相位为0，构建一个复数场
        # 注意：这里的相位是未知的，网络的核心任务就是去恢复它
        initial_phase = torch.zeros_like(amplitude)
        complex_field_diffraction = torch.complex(amplitude, initial_phase)

        # 模块一：傅里叶变换 (FFT)
        spectrum = torch.fft.fftshift(torch.fft.fft2(complex_field_diffraction))

        # 将复数频谱的实部和虚部作为两个通道输入到可学习模块
        spectrum_channels = torch.cat([spectrum.real, spectrum.imag], dim=1)

        # 模块二：可学习的逆向传播模块
        corrected_spectrum_channels = self.inverse_propagator(spectrum_channels)

        # 将修正后的双通道输出重新组合成复数频谱
        corrected_real, corrected_imag = torch.chunk(corrected_spectrum_channels, 2, dim=1)
        corrected_spectrum = torch.complex(corrected_real, corrected_imag)

        # 模块三：逆傅里叶变换 (IFFT)
        reconstructed_complex_field = torch.fft.ifft2(torch.fft.ifftshift(corrected_spectrum))

        # 我们通常只关心重构后物体的振幅/强度
        reconstructed_amplitude = reconstructed_complex_field.abs()

        # 模块四：空间域精炼模块
        refined_output = self.spatial_refiner(reconstructed_amplitude)

        return refined_output
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __res_block(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1), # pad 1 pixel on each side
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True), # inplace modify the input directly, without allocating any additional output
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            nn.BatchNorm2d(out_chan),
        )

    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv1_res = self.__res_block(64, 64)

        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv2_res = self.__res_block(128, 128)

        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv3_res = self.__res_block(256, 256)
        self.Conv_add_3 = conv_block(ch_in=256, ch_out=256)

        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv4_res = self.__res_block(512, 512)
        self.Conv_add_4 = conv_block(ch_in=512, ch_out=512)

        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up_conv_add_4 = conv_block(ch_in=256, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv_add_3 = conv_block(ch_in=128, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x = self.Conv1_res(x1)
        x += x1
        x = self.relu(x)
        x = self.Maxpool(x)

        x2 = self.Conv2(x)
        x = self.Conv2_res(x2)
        x += x2
        x = self.relu(x)
        x = self.Maxpool(x)

        x3 = self.Conv3(x)
        x3 = self.Conv_add_3(x3)
        x = self.Conv3_res(x3)
        x += x3
        x = self.relu(x)
        x = self.Maxpool(x)

        x4 = self.Conv4(x)
        x4 = self.Conv_add_4(x4)
        x = self.Conv4_res(x4)
        x += x4
        x = self.relu(x)
        x = self.Maxpool(x)

        x5 = self.Conv5(x)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.Up_conv_add_4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.Up_conv_add_3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AE_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AE_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=16)
        self.Conv5 = conv_block(ch_in=16, ch_out=1)
        # self.Conv6 = conv_block(ch_in=8, ch_out=1)

        # self.Up6 = up_conv(ch_in=1, ch_out=8)
        # self.Up_conv6 = conv_block(ch_in=8, ch_out=8)
        #
        self.Up5 = up_conv(ch_in=1, ch_out=16)
        self.Up_conv5 = conv_block(ch_in=16, ch_out=16)

        self.Up4 = up_conv(ch_in=16, ch_out=32)
        self.Up_conv4 = conv_block(ch_in=32, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        # self.Up5 = up_conv(ch_in=64, ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=512, ch_out=512)
        #
        # self.Up4 = up_conv(ch_in=512, ch_out=256)
        # self.Up_conv4 = conv_block(ch_in=256, ch_out=256)
        #
        # self.Up3 = up_conv(ch_in=256, ch_out=128)
        # self.Up_conv3 = conv_block(ch_in=128, ch_out=128)
        #
        # self.Up2 = up_conv(ch_in=128, ch_out=64)
        # self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def encoder(self, x):
        # encoding path  x --> 256*256*3 (196608)  x5 --> 16*16*1 (256)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # latent = self.Maxpool(x5)
        # latent = self.Conv6(latent)

        # return latent
        return x5

    def decoder(self, latent):
        # decoding + concat path
        # latent = self.Up6(latent)
        # d5 = self.Up_conv6(latent)
        #
        d5 = self.Up5(latent)
        # d5 = self.Up5(d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def forward(self, x):
        latent = self.encoder(x)
        d1 = self.decoder(latent)

        return d1


class Encoder_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Encoder_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=16)
        self.Conv5 = conv_block(ch_in=16, ch_out=8)
        self.Conv6 = conv_block(ch_in=8, ch_out=4)
        self.Conv7 = conv_block(ch_in=4, ch_out=1)

        latent_dim = 32 * 32 * 1  #
        self.mlp_1 = nn.Linear(latent_dim, latent_dim*2)
        self.mlp_2 = nn.Linear(latent_dim*2, latent_dim*2)
        self.mlp_3 = nn.Linear(latent_dim*2, 9)
        self.activation = nn.LeakyReLU()

    def encoder(self, x):
        # encoding path  x --> 256*256*3 (196608)  x5 --> 16*16*1 (256)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        # x3 = self.Maxpool(x2)
        # x3 = self.Conv3(x3)
        x3 = self.Conv3(x2)

        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)
        x4 = self.Conv4(x3)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        x5 = self.Conv5(x4)

        # latent = self.Maxpool(x5)
        x6 = self.Conv6(x5)
        x7 = self.Conv7(x6)

        # return latent
        return x7  # x5

    def head(self, latent):
        h = self.mlp_1(latent)
        h = self.activation(h)
        h = self.mlp_2(h)
        h = self.activation(h)
        h = self.mlp_3(h)

        return h

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.view(latent.size(0), -1)
        d1 = self.head(latent)

        return d1



class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



########################################################################################################################
# ------------------------> FOR Attention UNet
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from Attention.attend import Attend
from functools import partial
import math
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Transformer_UNet(nn.Module):
    def __init__(
        self,
        L=0,
        dim=64,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        encoder_only = False,
        output_hw = 3*3,
        use_resnet_branch=False,
        fusion_method='concat',  # 'concat', 'add', 'gate'
    ):
        super().__init__()


        # 保存双分支配置
        self.use_resnet_branch = use_resnet_branch
        self.fusion_method = fusion_method
        self.encoder_only = encoder_only

        #transformer初始化代码
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                # 下采样：最后阶段用卷积，其他用2倍下采样
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.resnet_up_to_32 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # ===================== 双分支修改从这里开始 =====================
        if self.use_resnet_branch:
            # 创建ResNet18分支 (修改第一层以适应单通道输入)
            self.resnet_branch = models.resnet18(weights = None)
            self.resnet_branch.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # 移除最后的池化和全连接层，保留特征提取部分
            self.resnet_branch = nn.Sequential(
                *list(self.resnet_branch.children())[:-2])

            # 自适应池化确保特征图尺寸匹配
            self.adapt_pool = nn.AdaptiveAvgPool2d(
                (self.downsample_factor, self.downsample_factor))

            # 根据融合方法创建融合模块
            if fusion_method == 'concat':
                # 简单concat + 卷积融合
                self.fusion = nn.Sequential(
                    nn.Conv2d(mid_dim + 512, mid_dim, kernel_size=1),
                    nn.ReLU(inplace=True)
                )
            elif fusion_method == 'add':
                # 特征相加融合
                self.fusion = nn.Sequential(
                    nn.Conv2d(512, mid_dim, kernel_size=1),
                    nn.ReLU(inplace=True)
                )
            elif fusion_method == 'gate':
                # 动态门控融合 (高级选项)
                self.fusion = DynamicGateFusion(mid_dim, 512, mid_dim)
            else:
                raise ValueError(f"未知的融合方法: {fusion_method}")
        # ===================== 双分支修改到这里结束 =====================


        if self.encoder_only:
            self.mid_block2 = block_klass(mid_dim, dim, time_emb_dim = time_dim)
        else:
            self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        if not self.encoder_only:
            for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
                is_last = ind == (len(in_out) - 1)
            
                attn_klass = FullAttention if layer_full_attn else LinearAttention
            
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                    attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
                ]))
            
            default_out_dim = channels * (1 if not learned_variance else 2)
            self.out_dim = default(out_dim, default_out_dim)
            
            self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
            self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        if self.encoder_only:
            latent_dim = dim*8*8  #
            self.mlp_1 = nn.Linear(latent_dim, latent_dim*2)
            self.mlp_2 = nn.Linear(latent_dim*2, latent_dim*2)
            self.mlp_3 = nn.Linear(latent_dim*2, output_hw*output_hw)
            self.activation = nn.LeakyReLU()

        self.L = L

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def head(self, latent):
        # print(f"Latent shape before flatten: {latent.shape}")
        h = self.mlp_1(latent)
        h = self.activation(h)
        h = self.mlp_2(h)
        h = self.activation(h)
        h = self.mlp_3(h)

        return h

    def default(self, val, d):
        return val if val is not None else d

    def cast_tuple(self, t, length=1):
        return t if isinstance(t, tuple) else (t,) * length

    def forward(self, x, time=None, x_self_cond = None):
        # print(f"\n原始尺寸:{x.shape}")
        raw_input = x[:, self.channels:] if self.self_condition else x
        # print(f"\nraw_input尺寸:{raw_input.shape}")
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'输入尺寸  {x.shape[-2:]} 需要能被{self.downsample_factor}整除'
        if time is None:
            time = self.L * torch.ones((x.shape[0],), device=x.device).long()

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []#储存跳跃连接特征
        # ===================== 原始Transformer分支 =====================
        for block1, block2, attn, downsample in self.downs:
            #第一个resnet块
            x = block1(x, t)
            h.append(x)
            # 第二个ResNet块
            x = block2(x, t)
            # 注意力机制（残差连接）
            x = attn(x) + x
            h.append(x) # 存储特征
            # 下采样
            x = downsample(x)
        # ===================== 双分支处理 =====================
        if self.use_resnet_branch:
            # 保存Transformer分支的特征
            trans_feat = x
            # print(f"\nResNet分支输出尺寸: {raw_input.shape}")  # 打印完整维度
            # 计算ResNet分支特征 (使用原始输入)
            resnet_feat = self.resnet_branch(x_self_cond if self.self_condition else raw_input)
            # print(f"\nResNet分支输出尺寸: {resnet_feat.shape}")  # 打印完整维度
            resnet_feat = self.adapt_pool(resnet_feat)

            # 特征融合
            if self.fusion_method == 'concat':
                # Concat融合
                # resnet_feat = self.resnet_up_to_32(resnet_feat)
                # print(f"\ntransformer尺寸{trans_feat.shape}")
                # print(f"\nresnet尺寸{resnet_feat.shape}")
                x = torch.cat([trans_feat, resnet_feat], dim=1)
                x = self.fusion(x)
            elif self.fusion_method == 'add':
                # 相加融合
                resnet_feat = self.fusion(resnet_feat)
                x = trans_feat + resnet_feat
            elif self.fusion_method == 'gate':
                # 门控融合
                x = self.fusion(trans_feat, resnet_feat)
        # ===================== 双分支处理结束 =====================

        # 中间层
        x = self.mid_block1(x, t)
        #注意力机制残差连接
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        #上采样
        if not self.encoder_only:
            for block1, block2, attn, upsample in self.ups:
                # 跳跃连接1（从编码器获取特征）
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t)
                # 跳跃连接2
                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t) # 第二个ResNet块
                # 注意力机制（残差连接）
                x = attn(x) + x
            
                x = upsample(x)
            # 最终跳跃连接（初始特征）
            x = torch.cat((x, r), dim = 1)
            
            x = self.final_res_block(x, t)# 最终ResNet块
            output = self.final_conv(x)# 输出卷积

        if self.encoder_only:
            latent = x.view(x.size(0), -1)
            output = self.head(latent)

        return output


# 动态门控融合模块 (可选)
class DynamicGateFusion(nn.Module):
    def __init__(self, channels1, channels2, out_channels):
        super().__init__()
        total_channels = channels1 + channels2

        # 门控权重生成器
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 特征融合
        self.fusion_conv = nn.Conv2d(total_channels, out_channels, kernel_size=1)

    def forward(self, feat1, feat2):
        # 确保特征图尺寸相同
        if feat1.shape[2:] != feat2.shape[2:]:
            feat2 = nn.functional.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)

        # 拼接特征
        combined = torch.cat([feat1, feat2], dim=1)

        # 生成门控权重 [B, 2, H, W]
        gate_weights = self.gate(combined)

        # 应用门控权重
        weighted_feat1 = gate_weights[:, 0:1] * feat1
        weighted_feat2 = gate_weights[:, 1:2] * feat2

        # 融合特征
        fused = torch.cat([weighted_feat1, weighted_feat2], dim=1)
        return self.fusion_conv(fused)