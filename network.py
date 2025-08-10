import torch

from torch.nn import init
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from configs.Sonet_configs import get_configs
config = get_configs()

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
# =================================================================================== #
# =================== 新增：transformer=============================== #
# =================================================================================== #
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).contiguous().chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads).contiguous(), qkv)

        # out = F.scaled_dot_product_attention(q, k, v)
        #
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # return self.to_out(out)
        #矩阵乘法
        dots = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale

        attn = self.attend(dots).contiguous()

        out = torch.matmul(attn, v.contiguous())
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
# =================================================================================== #


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
        self.encoder3 = conv_block(feature_channels * 2, feature_channels * 4) # 128
        self.pool3 = nn.MaxPool2d(2, 2)

        # --- Transformer 瓶颈层 ---
        # 1. 定义Transformer的参数
        bottleneck_dim = feature_channels * 4  # 进入Transformer的特征维度
        self.transformer_depth = 4  # Transformer Block的数量
        self.transformer_heads = 8  # 多头注意力的头数
        self.transformer_dim_head = 64  # 每个头的维度
        self.transformer_mlp_dim = bottleneck_dim * 2  # Transformer内部MLP的维度

        # 2. 位置编码：为展平的特征序列添加空间位置信息

        self.pos_embedding = nn.Parameter(torch.randn(1, 8 * 8, bottleneck_dim))

        # 3. 实例化Transformer
        self.transformer = Transformer(
            bottleneck_dim,
            self.transformer_depth,
            self.transformer_heads,
            self.transformer_dim_head,
            self.transformer_mlp_dim,
            dropout=0.1
        )

        self.upconv3 = up_conv(feature_channels * 4, feature_channels * 4)
        self.decoder3 = conv_block(feature_channels * 8, feature_channels * 4) # 输入是 upconv3 和 enc3 的拼接
        self.upconv2 = up_conv(feature_channels * 4, feature_channels * 2)
        self.decoder2 = conv_block(feature_channels * 4, feature_channels * 2)  # Takes concatenated input
        self.upconv1 = up_conv(feature_channels * 2, feature_channels)
        self.decoder1 = conv_block(feature_channels * 2, feature_channels)  # Takes concatenated input

        self.final_conv = nn.Conv2d(feature_channels, out_channels, 1)

    def forward(self, x):
        # Downsampling path
        enc1 = self.encoder1(x)# -> (B, f, H, W)
        enc2 = self.encoder2(self.pool1(enc1))# -> (B, 2f, H/2, W/2)
        enc3 = self.encoder3(self.pool2(enc2))# -> (B, 4f, H/4, W/4)
        pooled_enc3 = self.pool3(enc3)

        # --- Transformer 瓶颈层处理 ---
        # 1. 展平特征图: (B, C, H, W) -> (B, H*W, C)
        # B, C, H, W = enc3.shape
        # bottleneck_in = rearrange(enc3, 'b c h w -> b (h w) c')
        bottleneck_in = pooled_enc3.flatten(2).transpose(1, 2)


        # 2. 添加位置编码
        bottleneck_in += self.pos_embedding

        # 3. 通过Transformer处理
        transformer_out = self.transformer(bottleneck_in)

        # 4. 恢复特征图形状: (B, H*W, C) -> (B, C, H, W)
        # bottle = rearrange(transformer_out, 'b (h w) c -> b c h w', h=H, w=W)
        bottle = transformer_out.transpose(1, 2).view_as(pooled_enc3)


        up3 = self.upconv3(bottle)
        #print(f'env3:{enc3.shape},up3:{up3.shape},bottle:{bottle.shape}')
        dec3_in = torch.cat([up3, enc3], dim=1)
        dec3 = self.decoder3(dec3_in)  # 新增

        up2 = self.upconv2(dec3)
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




















