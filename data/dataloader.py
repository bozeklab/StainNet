import torch.nn as nn
import torchvision
from data.dataset import HER2Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from sklearn.utils import compute_sample_weight
import pandas as pd
import torch

SCORE_COL = "score-status-combined"

def get_transforms(finesize):
    print(f"Setting crop size to {finesize}")
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(finesize), 
        torchvision.transforms.ToTensor(),
    ])

    return transforms

def get_dataframe(args):
    df = pd.read_csv(args.csv_path)
    weights_dict = {0:1, 1:1, 2:2, 3:40, 4:40}
    weights = compute_sample_weight(weights_dict, df[SCORE_COL])

    train, test, train_weights, val_weights = train_test_split(df, weights, test_size=0.2)

    train["mode"] = "train"
    test["mode"] = "test"

    return pd.concat([train,test]), train_weights, val_weights

def get_dataloaders(args):
    train_tranforms = get_transforms(args.fineSize)
    val_tranforms = get_transforms(args.fineSize)

    df, train_weights, val_weights = get_dataframe(args)
    

    dataset_train = HER2Dataset(df, train_tranforms, args.dataroot, "train")
    dataset_val   = HER2Dataset(df, val_tranforms, args.dataroot, "test")
    sampler_train = WeightedRandomSampler(train_weights, args.epoch_len, replacement=True)
    sampler_val   = WeightedRandomSampler(val_weights  , args.val_len, replacement=True)
        
    dataloader_train = DataLoader(dataset_train, batch_size=args.batchSize,
                                  sampler=sampler_train, num_workers=args.nThreads)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batchSize,
                                  sampler=sampler_val  , num_workers=args.nThreads)
    return dataloader_train, dataloader_val