import os
import sys

#add path to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.BRTDataset import BRTDataset
import numpy as np

path = "6000_density_weighted"

dataset = BRTDataset(path, split="train", return_value_function=False)

dataset.compute_normalization_stats()