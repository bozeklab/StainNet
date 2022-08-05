from options.train_options import TrainOptions
from data.dataloader import get_dataloaders

def main(args):
    # args.csv_path = "result/csvs/stain_net_train.csv"
    train_loader, test_loader = get_dataloaders(args)

    a = next(iter(train_loader))
    print(a)


if __name__ == "__main__":
    parser = TrainOptions().parser
    args = parser.parse_args()

    main(args)
