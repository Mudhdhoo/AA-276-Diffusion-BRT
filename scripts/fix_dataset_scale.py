"""
Fix the issue of the data being in the range of 64x64x64 instead of 10x10xpi
"""


import numpy as np
from dataset.BRTDataset import BRTDataset
import os
from tqdm import tqdm

path = "../1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv"
dataset = BRTDataset(path)

for file in tqdm(dataset.point_cloud_files):
    point_cloud_path = os.path.join(path, file[0], file[1])
    point_cloud = np.load(point_cloud_path)
    point_cloud[:, 0] = point_cloud[:, 0]/64*10
    point_cloud[:, 1] = point_cloud[:, 1]/64*10
    point_cloud[:, 2] = point_cloud[:, 2]/64*2*np.pi - np.pi
    np.save(point_cloud_path, point_cloud)



