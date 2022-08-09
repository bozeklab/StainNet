from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--name", default="StainNet", type=str, help="name of the experiment.")
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir_net', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--fineSize', type=int, default=256, help='crop to this size')
        self.parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
        self.parser.add_argument('--test_freq', type=int, default=5, help='frequency of cal')
        self.parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for SGD')
        self.parser.add_argument('--epoch', type=int, default=300, help='how many epoch to train')
        self.parser.add_argument('--epoch_len', type=int, default=4000, help='how many samples per epoch')
        self.parser.add_argument('--val_len', type=int, default=500, help='how many samples per epoch')
        self.parser.add_argument('--continue_path', type=str, default=None, help="if you want to continue training provide path")

        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/stainGAN/', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument('--clearml_access_key', type=str, default="")
        self.parser.add_argument('--clearml_secret_key', type=str, default="")
        self.parser.add_argument('--clearml_task', type=str, default="Task")
        self.parser.add_argument('--clearml_project', type=str, default="Project")