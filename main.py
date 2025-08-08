import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 添加在文件最开头
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import numpy
import torch
from configs.Sonet_configs import get_configs



def main(config):


    # reproducibility
    seed = 3047
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    decay_ratio = 0.1
    decay_epoch = int(config.num_epochs * decay_ratio)
    config.model_path = os.path.join(config.dir_path, "models", config.special_save_folder_name)
    config.result_path = os.path.join(config.dir_path, "results", config.special_save_folder_name)
    config.log_path = os.path.join(config.dir_path, 'logs', config.special_save_folder_name)
    config.num_epochs_decay = decay_epoch

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # print(config)

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
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.,
                             exp_or_sim=config.exp_or_sim,
                             image_type=config.image_type,
                             config=config)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        #solver.test(pretrain_path=config.test_pretrained_model_path)
        solver.test(pretrain_path=os.path.join(
            config.model_path,
            config.result_model_path))
    elif config.mode == 'generate':
        solver.generate_test_result()


if __name__ == '__main__':

    config = get_configs()
    if config.test_pretrained_model_path is None:
        config.test_pretrained_model_path = os.path.join(
            config.model_path,
            config.special_save_folder_name,
            f'{config.model_type}.pkl')


    if config.mode == 'train':
        try:
            main(config)
        except Exception as e:
            raise e
    elif config.mode == 'test':
        test_fold = config.selected_test_fold
        config.batch_size = 1

        for fold in test_fold:
            config.selected_test_fold = [fold]
            config.result_path = config.save_path
            main(config)

    elif config.mode == 'generate':
            main(config)
