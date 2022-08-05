from options.test_options import TestOptions
from pytorch_lightning import Trainer 
from models import StainNet

def main(args):
    # predict_dataloader = 
    trainer = Trainer()
    model = StainNet(args).load_from_checkpoint(args.checkpoint_path)
    predictions = trainer.predict(model, dataloaders=predict_dataloader)

if __name__ == "__main__":
    parser = TrainOptions()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)