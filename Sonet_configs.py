import ml_collections
import os
import wget


def get_configs():
    cfg = ml_collections.ConfigDict()
    cfg.image_type = '8bits'  # '8bits' or '16bits'
    # dir_name = os.path.dirname(os.path.abspath(__file__))
    cfg.model_type = 'ASM_Net'
    #['UNet', 'CustomResNet34', 'CustomResNet18', 'Transformer_UNet','ASM_Net']
    NUM_L = 13
    cfg.exp_or_sim = 'exp'
    train_valid_folder = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    encoder_only = False
    output_hw = 64
    record_train_valid_name = f'enc_only_{encoder_only}_output_hw_{output_hw}'
    for f_name in train_valid_folder:
        record_train_valid_name += (f_name + '_')
    # 基础参数
    cfg.selected_train_valid_fold = train_valid_folder
    cfg.special_save_folder_name = f'{NUM_L}L_{cfg.exp_or_sim}_{cfg.model_type}_folders_{record_train_valid_name}'
    cfg.special_save_name = 'model'
    cfg.mode = 'train'  # train | test | generate
    cfg.image_size = 256
    cfg.t = 3
    cfg.valid_rate = 0.1

    cfg.dir_path = os.path.join("..", "ASM_main_result") #"D:\py\"
    #cfg.dir_path = os.path.dirname(os.path.abspath(__file__))
    cfg.save_path = f'save'

    cfg.selected_test_fold = ['SS', 'ORC']
    #物理一致性损失函数
    cfg.consistency_loss_weight = 0.1
    cfg.wavelength = 633e-9
    cfg.pixel_size = 41.7e-9
    cfg .propagation_distance= 10e-6
    #傅里叶损失函数
    cfg.edge_loss_weight = 0.1

    # --- 新增配置项 ---
    # 定义 PhaseHintNet 模型的根目录
    phase_hint_base_path = cfg.dir_path
    # 定义 PhaseHintNet 模型的完整路径
    cfg.phase_hint_model_path = os.path.join(phase_hint_base_path, 'PhaseHintNet', 'phase_hint_net_best.pkl')
    # --- 新增结束 ---


    # 训练相关参数
    cfg.img_ch = 1
    cfg.output_ch = 1
    cfg.num_epochs = 2
    cfg.num_epochs_decay = 30
    cfg.batch_size = 16
    cfg.num_workers = 8
    cfg.lr = 0.0001
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.wd = 1e-5
    cfg.augmentation_prob = 0.4
    cfg.log_step = 2
    cfg.val_step = 2
    cfg.focus_weight = 0.0
    cfg.focus_beta = 0.5

    # 模型参数

    cfg.model_type = 'ASM_Net'
    #['UNet', 'CustomResNet34', 'CustomResNet18', 'Transformer_UNet','ASM_Net']
    cfg.exp_or_sim = 'exp'
    cfg.rotate = True
    cfg.center_crop = False
    cfg.load_pretrain = False
    cfg.cuda_idx = 0

    # Transformer 参数
    cfg.L = NUM_L
    cfg.transformer_dim = 64
    cfg.encoder_only = encoder_only
    cfg.output_hw = output_hw

    # 路径设置

    cfg.model_path = os.path.join(cfg.dir_path, "models", cfg.special_save_folder_name)
    #cfg.result_path = 'results'
    cfg.train_path = '../data/train_valid/'
    cfg.valid_path = '../data/train_valid'
    cfg.test_path = '../data/test'
    #cfg.log_path = 'logs'
    cfg.log_path = os.path.join(cfg.dir_path, 'logs', cfg.special_save_folder_name)
    cfg.test_pretrained_model_path = None
    cfg.result_model_path = "ASM_Net_184.pkl"
    cfg.result_path = os.path.join(cfg.dir_path, "results", cfg.special_save_folder_name)

    return cfg