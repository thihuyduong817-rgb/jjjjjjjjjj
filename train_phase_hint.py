import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_loader
from network import PhaseHintNet  # 只从 network.py 导入 PhaseHintNet
from configs.Sonet_configs import get_configs


def train_phase_hint_net(config):
    """
    专门用于训练 PhaseHintNet 的主函数。
    """
    # 设置设备
    device = torch.device(f'cuda:{config.cuda_idx}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. 初始化 PhaseHintNet 模型 ---
    model = PhaseHintNet(img_ch=config.img_ch, output_ch=config.output_ch)
    model.to(device)

    # --- 2. 设置优化器和损失函数 ---
    # 使用 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    # 损失函数使用均方误差(MSE)，因为我们要预测的是连续的相位值
    criterion = nn.MSELoss()

    # --- 3. 加载数据 ---
    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob,
                              image_type=config.image_type,
                              exp_or_sim=config.exp_or_sim,
                              config=config)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=16,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.,
                              exp_or_sim=config.exp_or_sim,
                              image_type=config.image_type,
                              config=config)

    # --- 4. 设置模型保存路径 ---
    model_save_path = os.path.join(config.model_path, 'PhaseHintNet')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    best_model_path = os.path.join(model_save_path, 'phase_hint_net_best.pkl')

    # --- 5. 开始训练循环 ---
    best_valid_loss = float('inf')
    print("Starting training for PhaseHintNet...")

    for epoch in range(config.num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0

        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]")
        for images, GT, _, _ in train_pbar:
            images = images.to(device)
            GT = GT.to(device)

            # --- 核心步骤：从清晰图像GT生成代理真值相位 ---
            # a. 假设GT是纯振幅物体，构建复数场
            gt_complex = torch.complex(GT, torch.zeros_like(GT))
            # b. 傅里叶变换得到频谱
            gt_spectrum = torch.fft.fft2(gt_complex)
            # c. 提取相位角作为训练目标
            phase_ground_truth = torch.angle(gt_spectrum)

            # --- 模型预测 ---
            predicted_phase = model(images)

            # --- 计算损失 ---
            loss = criterion(predicted_phase, phase_ground_truth)

            # --- 反向传播和优化 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Valid]")
            for images, GT, _, _ in valid_pbar:
                images = images.to(device)
                GT = GT.to(device)

                # 同样地，从GT生成代理真值相位
                gt_complex = torch.complex(GT, torch.zeros_like(GT))
                gt_spectrum = torch.fft.fft2(gt_complex)
                phase_ground_truth = torch.angle(gt_spectrum)

                predicted_phase = model(images)
                loss = criterion(predicted_phase, phase_ground_truth)
                valid_loss += loss.item()
                valid_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_valid_loss = valid_loss / len(valid_loader)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} -> Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        # --- 保存最佳模型 ---
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved to {best_model_path} with validation loss: {best_valid_loss:.4f}")


if __name__ == '__main__':
    # 加载配置
    config = get_configs()

    # 为了训练PhaseHintNet，我们可以设置一个特定的模式
    # 你也可以在 Sonet_configs.py 中添加一个新的模式
    config.mode = 'train_phase_hint'

    # 运行训练函数
    train_phase_hint_net(config)

