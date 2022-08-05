from .base_options import BaseOptions


class InfereOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--one_sample', type=str, help='path to sample if only one sample should be checked')
        self.parser.add_argument('--checkpoints_dir', type=str, required=True, help='path to checkpoint')
        self.parser.add_argument('--results', type=str, default='./results/', help='save results there')
