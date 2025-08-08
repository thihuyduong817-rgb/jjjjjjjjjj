import torch

from torch.nn import init
import torch.nn as nn
from configs.Sonet_configs import get_configs
config = get_configs()
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
# =================== 新增：PhaseHintNet (相位提示网络) =============================== #
# =================================================================================== #
class PhaseHintNet(nn.Module):
    """
    一个轻量级的U-Net，用于从衍射强度图中预测一个初始相位。
    它的结构比主网络简单，旨在快速提供一个合理的相位猜测。
    """

    def __init__(self, img_ch=1, output_ch=1):
        super(PhaseHintNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(2, 2)

        self.Conv1 = conv_block(img_ch, 32)
        self.Conv2 = conv_block(32, 64)

        self.Up2 = up_conv(64, 32)
        self.Up_conv2 = conv_block(64, 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        # 使用 Tanh 激活函数将输出的相位限制在 [-pi, pi] 范围内 (乘以 pi)
        self.activation = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        # Decoder
        d2 = self.Up2(x2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # 将输出乘以pi，使其范围大致在[-pi, pi]
        return self.activation(d1) * torch.pi
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
    def __init__(self, img_ch=1, output_ch=1,use_phase_hint=True):
        super(ASM_Net, self).__init__()
        # ================ 修改：初始化PhaseHintNet ================ #
        self.use_phase_hint = use_phase_hint
        if self.use_phase_hint:
            self.phase_hint_net = PhaseHintNet(img_ch=img_ch, output_ch=output_ch)
            phase_hint_weights_path = config.phase_hint_model_path
            self.load_phase_hint_weights(phase_hint_weights_path)
            # 在训练ASM_Net时，通常冻结PhaseHintNet的权重
            for param in self.phase_hint_net.parameters():
                param.requires_grad = False
        # ======================================================== #
        # 模块二：可学习的逆向传播模块
        self.inverse_propagator = LearnableInversePropagator()

        # 模块四：空间域精炼模块
        self.spatial_refiner = nn.Sequential(
            conv_block(ch_in=img_ch, ch_out=64),
            conv_block(ch_in=64, ch_out=32),
            nn.Conv2d(32, output_ch, kernel_size=1)
        )

    # ================ 新增：加载PhaseHintNet预训练权重的方法 ================ #
    def load_phase_hint_weights(self, path):
        if self.use_phase_hint:
            self.phase_hint_net.load_state_dict(torch.load(path))
            print(f"Successfully loaded pre-trained weights for PhaseHintNet from {path}")
    # ====================================================================== #
    def forward(self, x):
        # 输入x是衍射图像的强度 (B, 1, H, W)
        # 我们假设输入强度是振幅的平方，先开方得到振幅
        amplitude = torch.sqrt(torch.clamp(x, min=1e-8))
        #print(f"config.phase_hint_model_path:{config.phase_hint_model_path}")
        # ================ 修改：使用PhaseHintNet或零相位 ================ #
        if self.use_phase_hint:
            # 在推理或主网络训练时，不计算梯度
            with torch.no_grad():
                initial_phase = self.phase_hint_net(x)
        else:
            # 原始方法：假设初始相位为0
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




















