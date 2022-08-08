import torch.nn as nn
import torchvision
from data.dataset import HER2Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import torch


def get_transforms(finesize):
    print(f"Setting crop size to {finesize}")
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(finesize), 
        torchvision.transforms.ToTensor(),
    ])

    return transforms

def get_dataframe(args):
    df = pd.read_csv(args.csv_path)
    train, test = train_test_split(df, test_size=0.2)

    train["mode"] = "train"
    test["mode"] = "test"

    return pd.concat([train,test])

def get_dataloaders(args):
    train_tranforms = get_transforms(args.fineSize)
    val_tranforms = get_transforms(args.fineSize)

    df = get_dataframe(args)

    dataset_train = HER2Dataset(df, train_tranforms, args.dataroot, "train")
    dataset_val   = HER2Dataset(df, val_tranforms, args.dataroot, "test")
    weigths = {"0":1, "1":2, "2":3, "3":30, "4": 50}
    train_weights = dataset_train.get_weights()
    val_weights   = dataset_val.get_weights()
    sampler_train = WeightedRandomSampler(train_weights, args.epoch_len, replacement=True)
    sampler_val   = WeightedRandomSampler(val_weights  , args.val_len, replacement=True)
        
    dataloader_train = DataLoader(dataset_train, batch_size=args.batchSize,
                                  sampler=sampler_train, num_workers=args.nThreads)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batchSize,
                                  sampler=sampler_val  , num_workers=args.nThreads)
    return dataloader_train, dataloader_val