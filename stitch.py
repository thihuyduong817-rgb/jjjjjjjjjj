import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from configs.Sonet_configs import get_configs
config = get_configs()
import os
# Define the image dimensions and patch size
def avg_stitch(
        images_array: np.ndarray,  # [66*66, 54, 54, 3]
        image_count=64,
        patch_per_image=3,
        stride=1) -> np.ndarray:
    output_size = image_count
    patch_size = images_array.shape[1] // patch_per_image
    # channel_size = images_array.shape[-1]
    # channel_size = 1

    # Initialize an array to hold the final stitched image
    # stitched_image = np.zeros((output_size * patch_size, output_size * patch_size, channel_size))
    stitched_image = np.zeros((output_size * patch_size, output_size * patch_size))


    # Function to get patch from an image for each color channel
    def get_patch(image, x_index, y_index, patch_size=18, patch_per_image=3):
        # Calculate the starting points for both dimensions
        mapping = {1:0, 0:1, -1:2}
        x_start = mapping[x_index] * patch_size
        y_start = mapping[y_index] * patch_size
        # return image[x_start:x_start + patch_size, y_start:y_start + patch_size, :]
        return image[x_start:x_start + patch_size, y_start:y_start + patch_size]

    # Function to handle edge and corner cases and average patches
    def average_patches(images, x, y):
        num_patches = 0
        # patch_sum = np.zeros((patch_size, patch_size, 3))
        patch_sum = np.zeros((patch_size, patch_size))

        # Define relative positions to extract patches from neighboring images
        relative_positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        idxes = []

        # Iterate over all possible relative positions
        for dx, dy in relative_positions:

            nx, ny = x + dx, y + dy
            if 0 < nx < 65 and 0 < ny < 65:
                image_index = (nx-1) * 66 + ny - 1
                idxes.append((image_index, (dx,dy)))
                patch = get_patch(images[image_index], dx, dy, patch_size)
                patch_sum += patch
                num_patches += 1

        if x == 0 and y == 2:
            print(num_patches, idxes)


        # Compute the average patch for each channel
        avg_patch = patch_sum / num_patches


        # 区别在这！！！ 如果是GT的话加这一步可以模拟SR的效果
        # # find the average value of avg_patch and set all the values to the average value
        # avg_patch_avg_value = np.mean(avg_patch)
        # avg_patch = np.full((patch_size, patch_size, 3), avg_patch_avg_value)


        return avg_patch

    # Loop over each patch position in the output image
    for i in range(output_size):
        for j in range(output_size):
            # Identify the corresponding image and patch within the image
            avg_patch = average_patches(images_array, i, j)

            # Calculate the position in the final stitched image
            x_start = i * patch_size
            y_start = j * patch_size

            # Place the averaged patch into the correct position in the stitched image
            # stitched_image[x_start:x_start + patch_size, y_start:y_start + patch_size,
            # :] = avg_patch
            stitched_image[x_start:x_start + patch_size, y_start:y_start + patch_size] = avg_patch

    return stitched_image
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if __name__ == "__main__":
    #拼接小图成为大图
    # Define the image dimensions and patch size
    image_count = 66
    patch_per_scan = 3
    stride = 1

    # TO COMMENT OUT:
    # Read the images in [count, height, width, channel] format
    # images = np.random.rand(66 * 66, 64, 64, 3)

    # the images are in the format as test_CustomResNet34_idx_{i}_SR.png
    # read in sequence and form an array
    a = ["SS","ORC"]
    choose = a[0]
    model_path_path = os.path.splitext(os.path.basename(os.path.join(
        config.model_path,
        config.result_model_path)))[0]
    path = os.path.join(config.result_path,config.model_type,
                         model_path_path,
                        f'test_result_{choose}')
    print(path)
    # path = '/raid/crp.dssi/volume_Kubernetes/Benquan/data_L/AI_Optics_result_10L/results_lr_0.0001_bs_1/Transformer_UNet/test_result_A1'
    #path = 'D:/py/OpticalNet_main_result/results/10L_exp_transformerUnet_folders_enc_only_True_output_hw_64A1_A2_A3_A4_A5_A6_/Transformer_UNet/test_result_'+choose

    save = 'D:/py/result_pic'
    use_avg = True
    image_selected =[]

    prefix_name = "test_"+config.model_type
    data_chose = 'SR_010'  # GT SR SR_010 SR_030 050
    for i in range(66*66):
        image_selected.append(f'{path}/{prefix_name}_idx_{i}_{data_chose}.png')

    # images = np.array([np.array(Image.open(image)) for image in image_selected])
    # resize the image to 54x54
    images = np.array([np.array(Image.open(image).resize((54, 54))) for image in image_selected])


    # Perform average stitching
    if use_avg:
        stitched_image = avg_stitch(images, image_count, patch_per_scan, stride)
    else:
        stitched_image = images

    # save the stitched image
    stitched_image = Image.fromarray((stitched_image).astype(np.uint8))

    plt.imshow(stitched_image, cmap='viridis')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(f'{save}/stitched_{use_avg}_image_{data_chose}_{choose}_{model_path_path}.png', bbox_inches='tight', pad_inches=0)


    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:40:38 2023

@author: wbq17

"""

