from options.train_options import TrainOptions
from pytorch_lightning import Trainer 
from models import StainNet
from utilities.seed import set_seeds
from StainGAN.models.models import create_model
from clearml import Task
from data.dataloader import get_dataloaders


def set_task_credentials(args):
    web_server = 'https://app.community.clear.ml'
    api_server = 'https://api.community.clear.ml'
    files_server = 'https://files.community.clear.ml'
    access_key = args.clearml_access_key
    secret_key = args.clearml_secret_key

    if len(secret_key) + len(access_key) > 0:
        print("SETTING CREDENTIALS")
        args.logger = True
        Task.set_credentials(web_host=web_server,
                            api_host=api_server,
                            files_host=files_server,
                            key=access_key,
                            secret=secret_key)
    else:
        args.logger = False

def main(args):
    stain_gan = create_model(args)
    if args.continue_path is not None:
        model = StainNet.load_from_checkpoint(args.continue_path)
    else:
        model = StainNet(args)

    model.add_stain_gan(stain_gan)
    model.has_teardown_None = False
    model.setup_dataloader(args)
    set_seeds(args.seed)
    
    if args.logger: 
        task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
    args.max_epochs = args.epoch
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model)


if __name__ == "__main__":
    parser = TrainOptions().parser
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_task_credentials(args)
    args.isTrain = False
    if args.gpus < 1:
        args.gpu_ids = []
    else: 
        args.gpu_ids = list(range(args.gpus))
        
    main(args)
