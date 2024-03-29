from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import disruptor.UprightNet.models.networks
from disruptor.UprightNet.options.test_options import TestOptions
import sys
from disruptor.UprightNet.data.data_loader import *
from disruptor.UprightNet.models.models import create_model
import random
from tensorboardX import SummaryWriter
from PIL import Image

def save_image_size(image_path, output_file):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get the dimensions
            width, height = img.size

            # Write dimensions to a text file
            with open(output_file, 'w') as f:
                f.write(f"{height} {width}")

            print(f"Image size saved to '{output_file}' successfully.")
    except Exception as e:
        print(f"Error: {e}")

def get_roll_pitch():
    # from disruptor.sdquery import save_encoded_image #TODO import properly
    import os
    upright_path = 'disruptor/UprightNet'
    os.chdir(upright_path)
    # save_encoded_image(original_image_bytes, os.path.join(upright_path, 'imgs/rgb/users.png')) #TODO save to the right directory
    # save_encoded_image(normal_image_bytes, os.path.join(upright_path, 'imgs/normal_pair/users.png')) #TODO save to the right directory
    save_image_size('imgs/rgb/users.png', 'imgs/precomputed_crop_hw/users.txt')
    EVAL_BATCH_SIZE = 8
    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    eval_list_path = 'paths.txt'
    eval_num_threads = 3
    test_data_loader = CreateInferenceDataLoader(opt, eval_list_path,
                                                    False, EVAL_BATCH_SIZE,
                                                    eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= InteriorNet Test #images = %d ========='%test_data_size)

    model = create_model(opt, _isTrain=False)
    model.switch_to_train()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    global_step = 0

    def infer(model, dataset, global_step):

        count = 0.0

        model.switch_to_eval()

        count = 0

        for i, data in enumerate(dataset):
            stacked_img = data[0]

            pred_cam_geo_unit, pred_up_geo_unit, pred_weights = model.infer_model(stacked_img)
            from disruptor.UprightNet.models.networks import JointLoss
            pred_roll, pred_pitch = JointLoss.compute_angle_from_pred(pred_cam_geo_unit, pred_up_geo_unit, pred_weights)
            os.chdir('../..')
            return pred_roll, pred_pitch

    return infer(model, test_dataset, global_step)