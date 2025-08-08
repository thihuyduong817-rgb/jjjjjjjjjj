import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from data_loader import get_loader
from network import PhaseHintNet
from configs.Sonet_configs import get_configs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_and_visualize(config):
    """
    加载训练好的 PhaseHintNet 模型，对测试集进行预测，
    并生成预测相位与GT相位的并排对比图。
    """
    # --- 1. 环境与设备设置 ---
    device = torch.device(f'cuda:{config.cuda_idx}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. 初始化模型并加载权重 ---
    model = PhaseHintNet(img_ch=config.img_ch, output_ch=1)
    model_path = os.path.join(config.model_path, 'PhaseHintNet', 'phase_hint_net_best.pkl')

    if not os.path.exists(model_path):
        print(f"错误: 在路径 {model_path} 未找到模型。请先运行 train_phase_hint.py 进行训练。")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"模型已从 {model_path} 加载。")

    # --- 3. 加载测试数据集 ---
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.,
                             exp_or_sim=config.exp_or_sim,
                             image_type=config.image_type,
                             config=config)

    # --- 4. 创建保存结果的目录 ---
    result_save_path = "D:\py\phase_picture"
    os.makedirs(result_save_path, exist_ok=True)
    print(f"对比图像将保存至: {result_save_path}")

    # --- 5. 遍历测试集并生成对比图 ---
    test_pbar = tqdm(test_loader, desc="正在测试 PhaseHintNet 并生成图像")
    with torch.no_grad():
        for i, (images, GT, image_paths, _) in enumerate(test_pbar):
            images = images.to(device)
            GT = GT.to(device)

            # --- 生成真值相位 (Ground Truth Phase) ---
            gt_complex = torch.complex(GT, torch.zeros_like(GT))
            gt_spectrum = torch.fft.fftshift(torch.fft.fft2(gt_complex))
            phase_ground_truth = torch.angle(gt_spectrum)

            # --- 生成预测相位 (Predicted Phase) ---
            predicted_phase = model(images)

            # --- 将Tensor转为Numpy数组用于可视化 ---
            gt_phase_np = phase_ground_truth.squeeze().cpu().numpy()
            pred_phase_np = predicted_phase.squeeze().cpu().numpy()

            # --- 绘制并保存对比图 ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 绘制原始衍射图
            original_image_np = images.squeeze().cpu().numpy()
            im0 = axes[0].imshow(original_image_np, cmap='gray')
            axes[0].set_title('Input Diffraction Image')
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.05)

            # 绘制真值相位
            im1 = axes[1].imshow(gt_phase_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
            axes[1].set_title('Ground Truth Phase (from GT)')
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.05)

            # 绘制预测相位
            im2 = axes[2].imshow(pred_phase_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
            axes[2].set_title('Predicted Phase')
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.05)

            plt.tight_layout()

            # 从原始路径获取文件名并保存
            original_filename = os.path.basename(image_paths[0])
            save_filename = f"comparison_{original_filename.split('.')[0]}.png"
            plt.savefig(os.path.join(result_save_path, save_filename))
            plt.close(fig)


if __name__ == '__main__':
    # 加载配置
    config = get_configs()
    # 运行测试和可视化函数
    test_and_visualize(config)

