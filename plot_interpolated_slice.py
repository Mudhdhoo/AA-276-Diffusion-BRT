import matplotlib.pyplot as plt
import numpy as np
import os
from models.diffusion_modules import BRTDiffusionModel
from dataset.BRTDataset import BRTDataset
import torch
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and model    
dataset_path = "/Users/johncao/Library/CloudStorage/GoogleDrive-johncao@stanford.edu/My Drive/AA276/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv"
sample_path = os.path.join(dataset_path, "sample_119")

val_func = np.load(os.path.join(sample_path, "value_function.npy"))[-1]
env_grid = np.load(os.path.join(sample_path, "environment_grid.npy"))
env_grid = torch.from_numpy(env_grid).unsqueeze(0)
point_cloud = np.load(os.path.join(sample_path, "point_cloud_0.npy"))

dataset = BRTDataset("../1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv", split="train", with_value=True)

model = BRTDiffusionModel(
    state_dim=dataset.state_dim,
    env_size=dataset.env_size,
    num_points=dataset.num_points,
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.008,
    device=device
)

# Load checkpoint
checkpoint_path = "checkpoints/lucky_moon_21/checkpoint_epoch_2000.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Explicitly set weights_only=False
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"Loaded checkpoint from {checkpoint_path}")

x_t = torch.randn(1, model.num_points, model.state_dim).to(model.device)

reverse_process = []
for t in tqdm(reversed(range(model.num_timesteps)), desc="Denoising"):
    with torch.no_grad():
        t_batch = torch.full((1,), t, device=model.device, dtype=torch.long)
        x_t = model.p_sample(x_t, t_batch, env_grid, guidance_scale=1.7)
    reverse_process.append(x_t.clone())

final_pc = reverse_process[-1]
final_pc = final_pc.squeeze(0).cpu().numpy()
final_pc = dataset.denormalize_points(final_pc)

np.save("final_pc.npy", final_pc)

# Interpolate point cloud
# Extract x, y, z coordinates and color values
x = final_pc[:, 0]
y = final_pc[:, 1] 
z = final_pc[:, 2]
colors = final_pc[:, 3]

resolution = 64  # Adjust this for finer/coarser interpolation
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()

# Add some padding to ensure all points are within the interpolation domain
padding = 0.1  # 10% padding
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

x_min -= padding * x_range
x_max += padding * x_range
y_min -= padding * y_range
y_max += padding * y_range
z_min -= padding * z_range
z_max += padding * z_range

print(f"Point cloud bounds:")
print(f"  X: [{x.min():.4f}, {x.max():.4f}] -> Grid: [{x_min:.4f}, {x_max:.4f}]")
print(f"  Y: [{y.min():.4f}, {y.max():.4f}] -> Grid: [{y_min:.4f}, {y_max:.4f}]")
print(f"  Z: [{z.min():.4f}, {z.max():.4f}] -> Grid: [{z_min:.4f}, {z_max:.4f}]")
print(f"  Values: [{colors.min():.4f}, {colors.max():.4f}]")

# Create 3D grid
x_grid = np.linspace(x_min, x_max, resolution)
y_grid = np.linspace(y_min, y_max, resolution)
z_grid = np.linspace(z_min, z_max, resolution)

X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

# Flatten grid for interpolation
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])
data_points = np.column_stack([x, y, z])

print(f"Grid resolution: {resolution}³ = {resolution**3:,} points")
print(f"Original points: {len(final_pc):,}")

# RBF interpolation with thin plate spline
print("Performing RBF interpolation (thin plate spline)...")

# Check for duplicate points which can cause issues
unique_points, unique_indices = np.unique(data_points, axis=0, return_index=True)
if len(unique_points) < len(data_points):
    print(f"Warning: Found {len(data_points) - len(unique_points)} duplicate points, using unique points only")
    data_points = unique_points
    colors = colors[unique_indices]

rbf_tps = RBFInterpolator(data_points, colors, kernel='thin_plate_spline', smoothing=0.001)
F_rbf_tps = rbf_tps(grid_points).reshape(X_grid.shape)

# Create single comparison plot
print("Creating comparison plot...")

# Choose Z slice
z_slice_idx = resolution // 2
z_value = z_grid[z_slice_idx]

# Get interpolated Z-slice
interp_slice = F_rbf_tps[:, :, z_slice_idx]

# Get ground truth Z-slice
gt_z_idx = val_func.shape[2] // 2
gt_slice = val_func[:, :, gt_z_idx]

# Create coordinate grids for plotting
x_coords_gt = np.linspace(x_min, x_max, val_func.shape[0])
y_coords_gt = np.linspace(y_min, y_max, val_func.shape[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Interpolated data
im1 = ax1.imshow(interp_slice.T, extent=[x_min, x_max, y_min, y_max], 
                 origin='lower', cmap='viridis', aspect='auto')
ax1.contour(X_grid[:, :, z_slice_idx], Y_grid[:, :, z_slice_idx], interp_slice,
           levels=[0], colors='red', linewidths=2)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(f'Interpolated Data (Z = {z_value:.2f})')
ax1.grid(True, alpha=0.3)
plt.colorbar(im1, ax=ax1, label='Interpolated Value')

# Right plot: Ground truth
im2 = ax2.imshow(gt_slice.T, extent=[x_min, x_max, y_min, y_max],
                 origin='lower', cmap='viridis', aspect='auto')
ax2.contour(x_coords_gt, y_coords_gt, gt_slice.T,
           levels=[0], colors='red', linewidths=2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Ground Truth (Z-slice idx {gt_z_idx})')
ax2.grid(True, alpha=0.3)
plt.colorbar(im2, ax=ax2, label='Ground Truth Value')

plt.tight_layout()
plt.show()

# Additional plot: Interpolated point cloud (negative values only)
print("Plotting interpolated point cloud (negative values only)...")

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# Left: 3D scatter plot of negative values from interpolated data
ax3_3d = fig2.add_subplot(121, projection='3d')
fig2.delaxes(ax3)

# Find all negative values in the interpolated data
neg_mask_3d = F_rbf_tps < 0
neg_indices = np.where(neg_mask_3d)
neg_x = X_grid[neg_indices]
neg_y = Y_grid[neg_indices]
neg_z = Z_grid[neg_indices]
neg_values = F_rbf_tps[neg_indices]

# 3D scatter plot of interpolated negative values
scatter_3d = ax3_3d.scatter(neg_x, neg_y, neg_z, c=neg_values, cmap='viridis', 
                           alpha=0.6, s=1)
ax3_3d.set_xlabel('X')
ax3_3d.set_ylabel('Y')
ax3_3d.set_zlabel('Z')
ax3_3d.set_title('Interpolated Diffusion Point Cloud')
plt.colorbar(scatter_3d, ax=ax3_3d, shrink=0.6, label='Value')

# Right: 3D scatter plot of negative values from ground truth
ax4_3d = fig2.add_subplot(122, projection='3d')
fig2.delaxes(ax4)

# Create coordinate grids for ground truth
Nx_gt, Ny_gt, Nz_gt = val_func.shape
x_gt = np.linspace(x_min, x_max, Nx_gt)
y_gt = np.linspace(y_min, y_max, Ny_gt)
z_gt = np.linspace(z_min, z_max, Nz_gt)
X_gt, Y_gt, Z_gt = np.meshgrid(x_gt, y_gt, z_gt, indexing='ij')

# Find negative values in ground truth
gt_neg_mask = val_func < 0
gt_neg_indices = np.where(gt_neg_mask)
gt_neg_x = X_gt[gt_neg_indices]
gt_neg_y = Y_gt[gt_neg_indices]
gt_neg_z = Z_gt[gt_neg_indices]
gt_neg_values = val_func[gt_neg_indices]

# 3D scatter plot of ground truth negative values
scatter_3d_gt = ax4_3d.scatter(gt_neg_x, gt_neg_y, gt_neg_z, c=gt_neg_values, 
                              cmap='viridis', alpha=0.6, s=1)
ax4_3d.set_xlabel('X')
ax4_3d.set_ylabel('Y')
ax4_3d.set_zlabel('Z')
ax4_3d.set_title('Ground Truth Value Function')
plt.colorbar(scatter_3d_gt, ax=ax4_3d, shrink=0.6, label='Value')

plt.tight_layout()
plt.show()

print(f"Interpolated slice shape: {interp_slice.shape}")
print(f"Ground truth slice shape: {gt_slice.shape}")
print(f"Interpolated value range: [{interp_slice.min():.4f}, {interp_slice.max():.4f}]")
print(f"Ground truth value range: [{gt_slice.min():.4f}, {gt_slice.max():.4f}]")

# Point cloud comparison
print("Plotting point cloud comparison...")

fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Generated point cloud
ax5_3d = fig3.add_subplot(121, projection='3d')
fig3.delaxes(ax5)

scatter_gen = ax5_3d.scatter(final_pc[:, 0], final_pc[:, 1], final_pc[:, 2], 
                            c=final_pc[:, 3], cmap='viridis', alpha=0.6, s=2)
ax5_3d.set_xlabel('X')
ax5_3d.set_ylabel('Y')
ax5_3d.set_zlabel('Z')
ax5_3d.set_title('Generated Point Cloud')
plt.colorbar(scatter_gen, ax=ax5_3d, shrink=0.6, label='Generated Value')

# Right: Ground truth point cloud
ax6_3d = fig3.add_subplot(122, projection='3d')
fig3.delaxes(ax6)

scatter_gt = ax6_3d.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                           c=point_cloud[:, 3], cmap='viridis', alpha=0.6, s=2)
ax6_3d.set_xlabel('X')
ax6_3d.set_ylabel('Y')
ax6_3d.set_zlabel('Z')
ax6_3d.set_title('Ground Truth Point Cloud')
plt.colorbar(scatter_gt, ax=ax6_3d, shrink=0.6, label='Ground Truth Value')

plt.tight_layout()
plt.show()

print(f"Generated point cloud shape: {final_pc.shape}")
print(f"Ground truth point cloud shape: {point_cloud.shape}")
print(f"Generated value range: [{final_pc[:, 3].min():.4f}, {final_pc[:, 3].max():.4f}]")
print(f"Ground truth value range: [{point_cloud[:, 3].min():.4f}, {point_cloud[:, 3].max():.4f}]")