import torch
import random 
import numpy as np 

def set_seeds(seed=42) -> None:
    """
    Set random seeds for reproduceability.
    See: https://pytorch.org/docs/stable/notes/randomness.html
         https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    """
    torch.manual_seed(seed) 
    random.seed(seed)
    np.random.seed(seed)