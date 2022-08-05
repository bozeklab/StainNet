import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def parse(self):
        if not self.initialized:
            self.initialized = True
        else:
            return self.opt 
        self.opt = self.parser.parse_args()

        return self.opt
