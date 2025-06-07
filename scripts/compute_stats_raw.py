import os
import numpy as np

path = "../6000_density_weighted"

mean = []

samples = os.listdir(path)
for sample in samples:
    sample_path = os.path.join(path, sample)
    point_clouds = os.listdir(sample_path)
    point_clouds = [f for f in point_clouds if f.startswith("point_cloud_")]
    point_clouds_mean = []
    for point_cloud in point_clouds:
        point_cloud_path = os.path.join(sample_path, point_cloud)
        point_cloud = np.load(point_cloud_path)
        point_clouds_mean.append(np.mean(point_cloud, axis=0))
    mean.append(np.mean(point_clouds_mean, axis=0))

global_mean = np.mean(mean, axis=0)

std = []
for sample in samples:
    sample_path = os.path.join(path, sample)
    point_clouds = os.listdir(sample_path)
    point_clouds = [f for f in point_clouds if f.startswith("point_cloud_")]
    point_clouds_std = []
    for point_cloud in point_clouds:
        point_cloud_path = os.path.join(sample_path, point_cloud)
        point_cloud = np.load(point_cloud_path)
        point_clouds_std.append(np.mean((point_cloud - global_mean) ** 2, axis=0))
    std.append(np.mean(point_clouds_std, axis=0))

global_std = np.sqrt(np.mean(std, axis=0))


print(f"Mean: {global_mean}")
print(f"Std: {global_std}")






