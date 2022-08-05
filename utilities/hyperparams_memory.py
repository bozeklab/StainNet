import pandas as pd 
import os

CSV_PATH = "hyperparameters_performance.csv"
COLUMNS = [
    "filename",
    "score",
    "name",
    "n_layer",
    "channels",
]

class HyperparametersMemory():
    def __init__(self):
        if not os.path.exists(CSV_PATH):
            df = pd.DataFrame({key: [] for key in COLUMNS})
            df.to_csv(CSV_PATH, index=False)
        self.df = pd.read_csv(CSV_PATH)

    def push_data(self, filename, data_dict):
        if self.filename_exists(filename):
            self.update_data(filename, data_dict)
        else:
            self.df.append(data_dict, ignore_index=True)
        
    def filename_exists(self, filename):
        return filename in self.df["filename"]
    
    def update_data(self, filename, data_dict):
        for key, value in data_dict.items():
            self.df[self.df["filename"] == filename][key] = value 

    