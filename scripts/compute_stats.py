import os
import sys

#add path to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.BRTDataset import BRTDataset
import numpy as np

path = "/Users/malte/AA-276-Diffusion-BRT/5000_1000_newest"

dataset = BRTDataset(path, split="train", return_value_function=False)

dataset.compute_normalization_stats()