import os
import numpy as np
import torchvision
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AE_Net, Encoder_Net, Transformer_UNet ,ASM_Net
from tqdm import tqdm
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from comp_pearson_corr import pearson_correlation
from Basic_Unet import UNet, CustomResNet34, CustomResNet18
# import wandb
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd
from configs.Sonet_configs import get_configs
config = get_configs()

import traceback
def f_beta_loss(preds, labels, beta=1, threshold=0.5):
    """
    计算F-beta损失函数
    参数:
        preds: 模型预测值
        labels: 真实标签
        beta: 控制精确率和召回率相对重要性的参数
        threshold: 将概率转换为二分类的阈值
    返回:
        F-beta损失值
    """
    epsilon = 1e-7  #  防止除零的小常数
    preds = (preds > threshold).float() # 根据阈值二值化预测结果

    # 计算真正例、假正例、假反例
    true_positives = torch.sum(preds * labels)
    false_positives = torch.sum(preds * (1 - labels))
    false_negatives = torch.sum((1 - preds) * labels)

    # 计算精确率和召回率
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)

    # 计算F-beta分数
    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (precision * recall) / \
                   ((beta_squared * precision) + recall + epsilon)
    # 返回F-beta损失的倒数(因为我们要最小化损失)
    f_beta_loss = 1. / (f_beta_score.mean() + epsilon)

    return f_beta_loss

# =================================================================================== #
# =================== 新增：物理一致性损失函数 ====================================== #
# =================================================================================== #
def forward_propagator(shape, dx, wavelength, z):
    """
    生成正向传播的传递函数 H (在频域中)。
    """
    B, C, H, W = shape
    k = 2 * np.pi / wavelength

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建频率坐标
    fx = torch.fft.fftfreq(W, d=dx).to(device)
    fy = torch.fft.fftfreq(H, d=dx).to(device)
    kx, ky = 2 * np.pi * fx, 2 * np.pi * fy
    KX, KY = torch.meshgrid(ky, kx, indexing='ij')

    # 计算 kz
    kz_sq = k ** 2 - KX ** 2 - KY ** 2
    kz = torch.sqrt(torch.complex(kz_sq.float(), torch.zeros_like(kz_sq).float()))

    # 传播函数 H
    H_propagator = torch.exp(1j * kz * z)
    return H_propagator.unsqueeze(0).unsqueeze(0)


def physical_consistency_loss(predicted_object, original_diffraction, config):
    """
    计算重构图像与原始衍射图之间的物理一致性。
    """
    # 假设重构出的物体没有相位（或相位为0）
    predicted_object_field = torch.complex(predicted_object, torch.zeros_like(predicted_object))

    # 1. 正向传播：FFT
    object_spectrum = torch.fft.fft2(predicted_object_field)

    # 2. 正向传播：乘以传播函数
    propagator = forward_propagator(
        shape=predicted_object.shape,
        dx=config.pixel_size,
        wavelength=config.wavelength,
        z=config.propagation_distance
    )
    propagated_spectrum = object_spectrum * propagator

    # 3. 正向传播：IFFT
    simulated_diffraction_field = torch.fft.ifft2(propagated_spectrum)

    # 4. 计算模拟衍射图的强度
    simulated_diffraction_intensity = simulated_diffraction_field.abs() ** 2

    # 归一化以进行比较
    sim_norm = (simulated_diffraction_intensity - simulated_diffraction_intensity.min()) / (
                simulated_diffraction_intensity.max() - simulated_diffraction_intensity.min() + 1e-8)
    orig_norm = (original_diffraction - original_diffraction.min()) / (
                original_diffraction.max() - original_diffraction.min() + 1e-8)

    # 5. 计算与原始衍射图强度的MSE损失
    loss = F.mse_loss(sim_norm, orig_norm)
    return loss

# =================================================================================== #
# =================== 新增：傅里叶边缘损失函数 (Fourier Edge Loss) ==================== #
# =================================================================================== #
def fourier_edge_loss(pred, target, cutoff_ratio=0.15):
    """
    计算预测图像和目标图像在频域高频分量上的差异。
    这会引导模型生成具有与真实图像相似边缘结构的结果。

    参数:
    - pred (Tensor): 模型的预测输出 (B, C, H, W)，值在 [0, 1] 区间。
    - target (Tensor): 真实标签图像 (B, C, H, W)。
    - cutoff_ratio (float): 截止频率比率。用于定义高通滤波器，滤除低频信息。
                           值越小，高通滤波越强。
    """
    # 1. 对预测和目标图像进行2D傅里叶变换
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))

    # 2. 将零频率分量移动到频谱中心，便于处理
    pred_fft_shifted = torch.fft.fftshift(pred_fft)
    target_fft_shifted = torch.fft.fftshift(target_fft)

    # 3. 创建一个高通滤波器掩码 (High-pass filter mask)
    B, C, H, W = pred.shape
    center_h, center_w = H // 2, W // 2

    # 根据截止频率比率计算半径
    radius_h = int(center_h * cutoff_ratio)
    radius_w = int(center_w * cutoff_ratio)

    mask = torch.ones_like(pred_fft_shifted, device=pred.device)
    # 在中心区域（低频区域）置为0
    mask[:, :, center_h - radius_h: center_h + radius_h, center_w - radius_w: center_w + radius_w] = 0

    # 4. 应用高通滤波器，仅保留边缘等高频信息
    pred_high_freq = pred_fft_shifted * mask
    target_high_freq = target_fft_shifted * mask

    # 5. 计算高频分量幅度的均方误差损失
    # 我们关心的是边缘的强度，而不是相位，所以使用幅值进行比较
    loss = F.mse_loss(torch.abs(pred_high_freq), torch.abs(target_high_freq))

    return loss

def concat_images(image_paths, output_path):
    """
    将多张图像水平拼接成一张图像并保存
    参数:
        image_paths: 图像路径列表
        output_path: 输出图像保存路径
    """
    images = [Image.open(x) for x in image_paths]# 打开所有图像
    width, height = images[0].size# 获取第一张图像的尺寸
    total_width = width * len(images)# 计算总宽度
    new_image = Image.new('RGBA', (total_width, height))# 创建新图像
    x_offset = 0
    # 将每张图像粘贴到新图像上
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += width
    new_image.save(output_path) # 保存拼接后的图像


class Solver(object):
    """
    主求解器类，负责模型的训练、验证和测试
    """
    def __init__(self, config, train_loader, valid_loader, test_loader):
        """
        初始化求解器
        参数:
            config: 配置对象，包含各种超参数和设置
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        self.log_path = config.log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.config = config
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None    # 主模型
        self.optimizer = None   # 优化器
        self.img_ch = config.img_ch # 输入图像的通道数
        self.output_ch = config.output_ch   # 输出通道数
        self.criterion = torch.nn.BCELoss() # 二元交叉熵损失
        self.augmentation_prob = config.augmentation_prob   # 数据增强概率

        # Hyper-parameters
        # 超参数
        self.lr = config.lr # 学习率
        self.beta1 = config.beta1   # Adam优化器的beta1
        self.beta2 = config.beta2   # Adam优化器的beta2

        self.focus_beta = config.focus_beta# F-beta损失中的beta参数

        # Training settings
        self.num_epochs = config.num_epochs # 总训练轮数
        self.num_epochs_decay = config.num_epochs_decay# 学习率衰减轮数
        self.batch_size = config.batch_size# 批大小

        # Step size
        self.log_step = config.log_step # 日志记录步长
        self.val_step = config.val_step # 验证步长

        # Path
        self.model_path = config.model_path  # 模型保存路径
        self.result_path = config.result_path   #结果保存路径
        self.mode = config.mode  # 运行模式

        # 使用指定GPU
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{config.cuda_idx}')

        self.model_type = config.model_type # 模型类型
        self.t = config.t   # R2U-Net中的时间步参数
        self.build_model()  # 构建模型
        # 学习率调度器(余弦退火)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)

    def build_model(self):

        """根据配置构建模型和优化器"""
        # 根据模型类型选择不同的网络结构
        if self.model_type == 'U_Net':
            # self.unet = U_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet = U_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)

            # ===================== try double ====================================== #
            # self.unet = self.unet.double()
            # ====================================================================== #
        elif self.model_type == 'AE_Net':
            self.unet = AE_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)

        elif self.model_type == 'Encoder_Net':
            self.unet = Encoder_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)
        # =================== 新增：构建ASM_Net模型 =================== #
        elif self.model_type == 'ASM_Net':
            self.unet = ASM_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)

        elif self.model_type == 'Transformer_UNet':
            self.unet = Transformer_UNet(L=self.config.L, dim=self.config.transformer_dim,
                                         encoder_only=self.config.encoder_only,
                                         output_hw=self.config.output_hw)

        elif self.model_type == 'UNet':
            self.unet = UNet(img_ch=self.config.img_ch, output_ch=self.config.output_ch)
            self.unet.initialize_weights()

        elif self.model_type == 'CustomResNet34':
            self.unet = CustomResNet34()

        elif self.model_type == 'CustomResNet18':
            self.unet = CustomResNet18()

        elif self.model_type == 'AE_Net_step1':
            self.unet = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            # 加载预训练模型
            self.unet.load_state_dict(torch.load(
                '/home/benquan/AE_Net_step1-250-0.0000-114-0.3052.pkl'))
        elif self.model_type == 'AE_Net_step2':
            self.unet = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet_mask = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet_mask.load_state_dict(torch.load(
                '/home/benquan/AE_Net_step1-250-0.0000-114-0.3052.pkl'))
            self.unet_mask.to(self.device)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=self.config.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=self.config.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=self.config.output_ch, t=self.t)
        # 初始化Adam优化器
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2], weight_decay=self.config.wd)
        # 将模型移动到指定设备
        self.unet.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        """打印网络结构和参数数量"""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        """将变量转换为张量并移动到CPU"""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        """更新学习率(未使用)"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """重置梯度"""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        """计算准确率(未完成)"""
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        """将张量转换为图像"""
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img



    def train(self):
        # ====================================== Training ===========================================#
        # ===========================================================================================#

        special_save_name = self.config.special_save_name
        unet_path = os.path.join(
            self.model_path,
            f'{self.model_type}.pkl'
        )


        #   创建日志目录
        log_dir = os.path.join(self.log_path, 'train_valid_records')
        os.makedirs(log_dir, exist_ok=True)

        print("[DEBUG] Final log_dir:", log_dir)
        print("log_dir:", log_dir)
        print("Length:", len(log_dir))

        writer = SummaryWriter(log_dir)

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            print('No training is executed.')

        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            best_epoch = 0
            best_unet_path = []
            train_log = []
            valid_log = []
            for epoch in range(self.num_epochs):
                # 训练阶段
                self.unet.train(True)
                epoch_loss = 0  #累计损失
                epoch_bce_loss = 0
                epoch_edge_loss = 0
                epoch_focus_loss = 0    # F-beta损失
                epoch_phys_loss = 0 # 物理一致性损失

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Starting epoch {epoch + 1} with learning rate: {current_lr}')

                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.train_loader)):
                    # 将数据移动到指定设备
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    # 前向传播
                    SR = self.unet(images)
                    SR_probs = F.sigmoid(SR)

                    # =================== 修改：计算混合损失 (Hybrid Loss) =================== #
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    # 计算损失
                    bce_loss = self.criterion(SR_flat, GT_flat)  # base loss


                    # 2. 物理一致性损失 (Physical Consistency Loss)
                    phys_loss = 0
                    if self.config.consistency_loss_weight > 0:
                        phys_loss = physical_consistency_loss(SR_probs, images, self.config)

                    # 3. 加权求和得到最终损失
                    loss = bce_loss + self.config.consistency_loss_weight * phys_loss

                    # ======================================================================= #

                    # ======================================================================= #
                    # 2. 新增的傅里叶边缘损失
                    # 注意：Transformer_UNet在encoder_only模式下输出的是向量，需要reshape
                    if self.config.model_type == 'Transformer_UNet' and self.config.encoder_only:
                        hw = int(np.sqrt(SR_probs.shape[-1]))
                        SR_probs_reshaped = SR_probs.view(SR_probs.size(0), 1, hw, hw)
                        edge_loss = fourier_edge_loss(SR_probs_reshaped, GT)
                    else:  # 其他模型直接输出图像
                        edge_loss = fourier_edge_loss(SR_probs, GT)
                    # 3. 加权求和得到最终损失
                    loss = bce_loss + self.config.edge_loss_weight * edge_loss
                    epoch_edge_loss += edge_loss.item()
                    # ======================================================================= #

                    epoch_bce_loss += bce_loss.item()

                    epoch_loss += loss.item()
                    if self.config.consistency_loss_weight > 0:
                        epoch_phys_loss += phys_loss.item()
                    # epoch_focus_loss += focus_loss.item()

                    # 反向传播
                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                    # 计算各种评估指标
                    acc += get_accuracy(SR_probs, GT)
                    SE += get_sensitivity(SR_probs, GT)
                    SP += get_specificity(SR_probs, GT)
                    PC += get_precision(SR_probs, GT)
                    F1 += get_F1(SR_probs, GT)
                    JS += get_JS(SR_probs, GT)
                    DC += get_DC(SR_probs, GT)
                    # length += images.size(0)
                    length += 1

                # break
                #计算平均指标
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                unet_score = 0.4 * DC + 0.3 * F1 + 0.2 * JS + 0.1 * SE
                # 计算平均损失
                # ===================  log in wandb ============ #
                total_loss = epoch_loss / length
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Finished epoch {epoch + 1}, new learning rate: {current_lr}')


                # Print the log info
                print(
                    'Epoch [%d/%d], Total_Loss: %.4f, Focus_loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f,Score: %.4f' % (
                        epoch + 1, self.num_epochs, \
                        epoch_loss / length, epoch_focus_loss / length, \
                        acc, SE, SP, PC, F1, JS, DC , unet_score))

                train_log.append({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss / length,
                    'train_acc': acc,
                    'train_SE': SE,
                    'train_SP': SP,
                    'train_PC': PC,
                    'train_F1': F1,
                    'train_JS': JS,
                    'train_DC': DC,
                    'train_score' :  0.4 * DC + 0.3 * F1 + 0.2 * JS + 0.1 * SE,
                    'lr': current_lr
                })
                writer.add_scalar('Train/Total_Loss', epoch_loss / length, epoch)
                writer.add_scalar('Train/BCE_Loss', epoch_bce_loss / length, epoch)
                writer.add_scalar('Train/Edge_Loss', epoch_edge_loss / length, epoch)
                writer.add_scalar('Train/SE', SE, epoch)
                writer.add_scalar('Train/Accuracy', acc, epoch)


                # ===================================== 验证 ====================================#
                with torch.no_grad():
                    self.unet.train(False)
                    self.unet.eval()

                    # 确保所有Dropout层都被禁用
                    for module in self.unet.modules():
                        if isinstance(module, nn.Dropout):
                            assert module.p == 0 or not module.training, "Dropout should be disabled in eval mode"
                    #重置评估指标
                    acc = 0.  # Accuracy
                    SE = 0.  # Sensitivity (Recall)
                    SP = 0.  # Specificity
                    PC = 0.  # Precision
                    F1 = 0.  # F1 Score
                    JS = 0.  # Jaccard Similarity
                    DC = 0.  # Dice Coefficient
                    length = 0
                    #验证集评估
                    for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.valid_loader)):
                        images = images.to(self.device)
                        GT = GT.to(self.device)
                        SR = F.sigmoid(self.unet(images))
                        # 计算各种评估指标
                        acc += get_accuracy(SR, GT)
                        SE += get_sensitivity(SR, GT)
                        SP += get_specificity(SR, GT)
                        PC += get_precision(SR, GT)
                        F1 += get_F1(SR, GT)
                        JS += get_JS(SR, GT)
                        DC += get_DC(SR, GT)

                        # length += images.size(0)
                        length += 1
                    # break
                #计算评估指标
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length

                unet_score = 0.4 * DC + 0.3 * F1 + 0.2 * JS + 0.1 * SE
                # unet_score = JS + DC
                #unet_score = acc
                #unet_score = SE

                print(
                    '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f，score: %.4f' % (
                        acc, SE, SP, PC, F1, JS, DC,unet_score))
                valid_log.append({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss / length,
                    'valid_acc': acc,
                    'valid_SE': SE,
                    'valid_SP': SP,
                    'valid_PC': PC,
                    'valid_F1': F1,
                    'valid_JS': JS,
                    'valid_DC': DC,
                    'valid_score': 0.4 * DC + 0.3 * F1 + 0.2 * JS + 0.1 * SE,
                    'lr': current_lr
                })
                if (epoch + 1) % 20 == 0 or (epoch + 1) == self.num_epochs:
                    df = pd.DataFrame(valid_log)
                    log_path = os.path.join(self.log_path, 'valid_log.xlsx')
                    df.to_excel(log_path, index=False)
                    df = pd.DataFrame(train_log)
                    log_path = os.path.join(self.log_path, 'training_log.xlsx')
                    df.to_excel(log_path, index=False)
                    print(f'Saved best model {best_unet_path}_{best_epoch}')

                writer.add_scalar('Valid_acc', acc, epoch)
                writer.add_scalar('Valid_SE', SE, epoch)


                # use threshold to preprocess SR img
                # # 使用不同阈值生成二值化预测结果
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)

                # Save Best U-Net model
                if unet_score >= best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    best_unet_path = os.path.join(
                        self.model_path,
                        f'{self.model_type}_best_epoch.pkl'
                    )
                    torch.save(best_unet, best_unet_path)
                if (epoch + 1) % 5== 0:
                    unet_path_30 = os.path.join(
                        self.model_path,
                        f'{self.model_type}_{epoch}.pkl'
                    )
                    print(unet_path_30)
                    torch.save(self.unet.state_dict(), unet_path_30)
                   # torch.save(best_unet, unet_path)

            final_model_path = os.path.join(
                self.model_path,
                f'{self.model_type}_final_epoch_{self.num_epochs}.pkl'
            )
            torch.save(self.unet.state_dict(), final_model_path)
            print(f'Saved final model (epoch {self.num_epochs}) to {final_model_path}')
            print(f'Saved best model {best_unet_path}_{best_epoch}')

        writer.close()

    def generate_test_result(self):
        """生成验证集预测结果"""
        unet_path = os.path.join(self.model_path, config.result_model_path)
        print(unet_path)
        if not os.path.isfile(unet_path):
            print(f"模型文件不存在: {unet_path}")
            return
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

            # =================================== generate test results one by one ==================================#
            with torch.no_grad():
                self.unet.train(False)
                self.unet.eval()

                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.valid_loader)):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = F.sigmoid(self.unet(images))

                    # # use threshold to preprocess SR img
                    SR_050 = torch.where(SR > 0.5, 1., 0.)

                    torchvision.utils.save_image(SR_050.data.cpu(),
                                                 os.path.join(self.result_path,
                                                              f'{gt_paths[17:-4]}_test.tif'))



    def tensor_to_3x3_image(self, tensor):
        """
        将张量转换为3x3图像
        参数:
            tensor: 输入张量
        返回:
            PIL Image对象
        """
        # Ensure the tensor is on CPU and detached from the computation graph
        # 确保张量在CPU上且脱离计算图
        tensor = tensor.cpu().detach()

        # Reshape the tensor to 3x3
        # hw = int(np.sqrt(tensor.shape[-1]))
        # img_array = tensor.view(hw, hw).numpy()
        # 处理不同维度的输入
        if tensor.dim() == 4:  # [batch, channel, H, W]
            img_array = tensor[0, 0].numpy()  # 取第一个样本的第一个通道
        elif tensor.dim() == 2:  # [batch, features]
            hw = int(np.sqrt(tensor.numel() // tensor.shape[0]))
            img_array = tensor[0].view(hw, hw).numpy()
        else:  # 其他情况直接展平
            total_pixels = tensor.numel()
            hw = int(np.sqrt(total_pixels))
            img_array = tensor.view(hw, hw).numpy()
        img_array = (img_array*255).astype(np.uint8)
        # 创建PIL图像
        img = Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale

        # 调整大小以便更好显示(可选)
        img = img.resize((64, 64), Image.NEAREST)

        return img

    def test(self, pretrain_path=None):
        """测试模型性能"""
        unet_path = pretrain_path

        model_filename = os.path.splitext(os.path.basename(unet_path))[0]
        # 创建测试结果保存路径
        self.test_result_path = os.path.join(self.result_path,
                                             model_filename,
                                             f'test_result_{self.config.selected_test_fold[0]}')

        print(
            f'test on fold {self.config.selected_test_fold[0]} and save results to {self.test_result_path}')
        if not os.path.exists(self.test_result_path):
            os.makedirs(self.test_result_path)
        # 加载预训练模型
        print(config.dir_path)
        test_test_path = os.path.join(config.dir_path,unet_path)

        self.unet.load_state_dict(torch.load(test_test_path),strict=False)
        print(f'{self.model_type} is Successfully Loaded from {test_test_path}')
        # print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

        # ===================================== Testing ====================================#
        with torch.no_grad():
            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            epoch_loss = 0.
            auc_roc = 0.
            length = 0
            CR = 0.  # Pearson correlation

            accs = []
            SEs = []
            PCs = []
            F1s = []
            auc_rocs = []
            CRs = []

            # create a dictionary to save the results of this subfolder
            correlation_result_dict = {
                '2 nanoholes': [],
                '3 nanoholes': [],
                '4 nanoholes': [],
                '5 nanoholes': [],
                '6 nanoholes': [],
                '7 nanoholes': [],
                '8 nanoholes': [],
                '9 nanoholes': [],
                '10 nanoholes': [],
            }

            for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.test_loader)):
                images = images.to(self.device)
                GT = GT.to(self.device)
                predict_image = self.unet(images)
                # predict_image_flatten = predict_image.flatten()

                SR = torch.sigmoid(predict_image)

                SR_flat = SR.view(SR.size(0), -1)

                GT_flat = GT.view(GT.size(0), -1)
                loss = self.criterion(SR_flat, GT_flat)
                epoch_loss += loss.item()

                point_acc = get_accuracy(SR, GT)
                point_SE = get_sensitivity(SR, GT)
                point_SP = get_specificity(SR, GT)
                point_PC = get_precision(SR, GT)
                point_F1 = get_F1(SR, GT)
                point_JS = get_JS(SR, GT)
                point_DC = get_DC(SR, GT)
                # GT_binary = np.round(GT.flatten().cpu().numpy())
                # point_auc_roc = roc_auc_score(y_true=GT_binary.flatten(),
                #                               y_score=SR.flatten().cpu().numpy())
                # point_CR = pearson_correlation(GT_flat, SR_flat)

                acc += point_acc
                SE += point_SE
                SP += point_SP
                PC += point_PC
                F1 += point_F1
                JS += point_JS
                DC += point_DC
                # auc_roc += point_auc_roc
                # CR += point_CR

                accs.append(point_acc)
                SEs.append(point_SE)
                PCs.append(point_PC)
                F1s.append(point_F1)
                # auc_rocs.append(point_auc_roc)
                # CRs.append(point_CR)

                # length += images.size(0)
                length += 1

                # use threshold to preprocess SR img
                # 使用不同阈值生成二值化预测
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)

                # 保存预测结果图像
                self.tensor_to_3x3_image(SR).save(os.path.join(self.test_result_path,

                                                               'test_%s_idx_%d_SR.png' % (
                                                                   self.model_type, i)))
                self.tensor_to_3x3_image(SR_075).save(os.path.join(self.test_result_path,

                                                                   'test_%s_idx_%d_SR_075.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(SR_050).save(os.path.join(self.test_result_path,

                                                                   'test_%s_idx_%d_SR_050.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(SR_030).save(os.path.join(self.test_result_path,

                                                                   'test_%s_idx_%d_SR_030.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(SR_010).save(os.path.join(self.test_result_path,

                                                                   'test_%s_idx_%d_SR_010.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(GT).save(os.path.join(self.test_result_path,

                                                               'test_%s_idx_%d_GT.png' % (
                                                                   self.model_type, i)))
                # 计算不同阈值下的敏感度
                output_list = (SR_075, SR_050, SR_030, SR_010)
                output_SEs = []

                # iterate over all the output images and compute the SE
                for output in output_list:
                    se = get_sensitivity_no_threshold(output, GT)
                    output_SEs.append(se)
                image = cv2.imread(os.path.join(self.test_result_path,
                                                'test_%s_idx_%d_SR_050.png' % (self.model_type, i)))
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_gray = cv2.resize(image_gray, (512, 512))
                gt = cv2.imread(GT_path[0])
                gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        epoch_loss = epoch_loss / length
        auc_roc = auc_roc / length
        CR = CR / length
        # unet_score = JS + DC
        unet_score = acc

        print(
            '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, AUROC: %.4f, Pearson: %.4f' % (
                epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR,))

        with open(os.path.join(self.test_result_path, 'metric_result_record.txt'), 'w') as f:
            f.write(
                '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, auroc: %.4f, Pearson: %.4f' % (
                    epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR))
