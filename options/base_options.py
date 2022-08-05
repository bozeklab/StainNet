import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

        # self.parser.add_argument("--source_root_train", default="dataset/Cytopathology/train/trainA", type=str, help="path to source images for training")
        # self.parser.add_argument("--gt_root_train", default="dataset/Cytopathology/train/trainB", type=str, help="path to ground truth images for training")
        # self.parser.add_argument("--source_root_test", default="dataset/Cytopathology/test/testA", type=str, help="path to source images for test")
        # self.parser.add_argument("--gt_root_test", default="dataset/Cytopathology/test/testB", type=str, help="path to ground truth images for test")
        self.parser.add_argument('--input_nc_net', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc_net', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--channels', type=int, default=32, help='# of channels in StainNet')
        self.parser.add_argument('--n_layer', type=int, default=3, help='# of layers in StainNet')
        self.parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
        self.parser.add_argument('--csv_path', type=str, required=True, help='path to images metadata')

        self.parser.add_argument('--dataroot', type=str, required=True, help='Directory containing images')
        self.parser.add_argument('--seed', type=int, default=42)

    def parse(self):
        if not self.initialized:
            self.initialized = True
        else:
            return self.opt 
        self.opt = self.parser.parse_args()

        return self.opt
