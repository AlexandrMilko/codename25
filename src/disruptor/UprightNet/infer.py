from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.test_options import TestOptions
import sys
from data.data_loader import *
from models.models import create_model
import random
from tensorboardX import SummaryWriter

def get_roll_pitch(original_image_bytes, normal_image_bytes):
    upright_path = 'disruptor/UprightNet'
    from disruptor.sdquery import save_encoded_image #TODO import properly
    import os
    save_encoded_image(original_image_bytes, os.path.join(upright_path, 'imgs/rgb/users.png')) #TODO save to the right directory
    save_encoded_image(normal_image_bytes, os.path.join(upright_path, 'imgs/normal_pair/users.png')) #TODO save to the right directory
    EVAL_BATCH_SIZE = 8
    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    eval_list_path = upright_path + 'paths.txt'
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
            from models.networks import JointLoss
            pred_roll, pred_pitch = JointLoss.compute_angle_from_pred(pred_cam_geo_unit, pred_up_geo_unit, pred_weights)
            return pred_roll, pred_pitch

    return infer(model, test_dataset, global_step)