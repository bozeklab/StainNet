import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from sklearn.utils import compute_sample_weight

FILENAME_COL = "filename"
SCORE_COL = "score-status-combined"

class HER2Dataset(Dataset):
    def __init__(self, df, transforms, dataroot, mode):
        self.df = df
        self.dataroot = dataroot
        self.transforms = transforms
        if mode is not None:
            self.df = self.df[self.df["mode"] == mode]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_row = self.df.iloc[idx]
        path = os.path.join(self.dataroot, img_row[FILENAME_COL])
        img_orig = Image.open(path).convert('RGB')

        return self.transforms(img_orig)

    def get_weights(self, class_weight="balanced"):
        """Returns the HER2 score calculated as weigths for WeightedRandomSampler"""
        return compute_sample_weight(class_weight, self.df[SCORE_COL])
