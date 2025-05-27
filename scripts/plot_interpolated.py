import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

data_file = 'point_cloud_0.npy'

# Load the point cloud data
point_cloud = np.load(data_file)

print(f"Point cloud shape: {point_cloud.shape}")
print(f"Data type: {point_cloud.dtype}")
print(f"First few points:\n{point_cloud[:5]}")

# Extract x, y, z coordinates and color values
x = point_cloud[:, 0]
y = point_cloud[:, 1] 
z = point_cloud[:, 2]
colors = point_cloud[:, 3]

print(f"\nCoordinate ranges:")
print(f"X: [{x.min():.3f}, {x.max():.3f}]")
print(f"Y: [{y.min():.3f}, {y.max():.3f}]")
print(f"Z: [{z.min():.3f}, {z.max():.3f}]")
print(f"Colors: [{colors.min():.3f}, {colors.max():.3f}]")

# Create 3D visualization of original point cloud
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot with color mapping
scatter = ax.scatter(x, y, z, c=colors, cmap='Spectral', s=3, alpha=1)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
cbar.set_label('Color Value', rotation=270, labelpad=15)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Original Point Cloud Visualization\n(Colored by value)', fontsize=14)

# Improve the view
ax.view_init(elev=20, azim=45)

# Make the plot look better
plt.tight_layout()
plt.show()

# Create XY projection visualization of original data
fig, ax = plt.subplots(figsize=(10, 8))

# XY projection
scatter = ax.scatter(x, y, c=colors, cmap='Spectral', s=3, alpha=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Original Point Cloud XY Projection', fontsize=14)
ax.set_aspect('equal')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Color Value', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

# Now create interpolated function f(x,y,z)
print("\nCreating interpolated function...")

# Define grid resolution
resolution = 64  # Adjust this for finer/coarser interpolation
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()

# Create 3D grid
x_grid = np.linspace(x_min, x_max, resolution)
y_grid = np.linspace(y_min, y_max, resolution)
z_grid = np.linspace(z_min, z_max, resolution)

X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

# Flatten grid for interpolation
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

# Use nearest neighbor interpolation for 3D data (more robust than linear for scattered 3D data)
print("Performing 3D interpolation...")
tree = cKDTree(np.column_stack([x, y, z]))
distances, indices = tree.query(grid_points, k=1)

# Get interpolated values
interpolated_values = colors[indices]

# Reshape back to 3D grid
F_grid = interpolated_values.reshape(X_grid.shape)

print(f"Interpolated grid shape: {F_grid.shape}")
print(f"Interpolated values range: [{F_grid.min():.3f}, {F_grid.max():.3f}]")

# Create side-by-side visualization: original point cloud and thresholded grid
fig = plt.figure(figsize=(16, 8))

# Plot 1: Original point cloud
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(x, y, z, c=colors, cmap='Spectral', s=3, alpha=1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Point Cloud', fontsize=14)
ax1.view_init(elev=20, azim=45)

# Add colorbar for original point cloud
cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20)
cbar1.set_label('Color Value', rotation=270, labelpad=15)

# Plot 2: Thresholded grid (values < 0 only)
ax2 = fig.add_subplot(122, projection='3d')

# Apply threshold: only show values less than 0
threshold_mask = F_grid < 0
X_thresh = X_grid[threshold_mask]
Y_thresh = Y_grid[threshold_mask]
Z_thresh = Z_grid[threshold_mask]
F_thresh = F_grid[threshold_mask]

print(f"Number of grid points with values < 0: {len(F_thresh)} out of {F_grid.size}")
print(f"Percentage of grid with values < 0: {100 * len(F_thresh) / F_grid.size:.1f}%")

if len(F_thresh) > 0:
    scatter2 = ax2.scatter(X_thresh, Y_thresh, Z_thresh, c=F_thresh, 
                          cmap='Spectral', s=20, alpha=1) 
    
    # Add colorbar for thresholded data
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20)
    cbar2.set_label('Function Value (< 0)', rotation=270, labelpad=15)
else:
    print("No grid points have values < 0")

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title(f'Thresholded Grid (values < 0)\nResolution: {resolution}³', fontsize=14)
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# Print statistics
print(f"\nStatistics:")
print(f"Grid resolution: {resolution}³ = {resolution**3:,} points")
print(f"Original points: {len(point_cloud):,}")
print(f"Interpolated function range: [{F_grid.min():.3f}, {F_grid.max():.3f}]")
print(f"Original data range: [{colors.min():.3f}, {colors.max():.3f}]")
if len(F_thresh) > 0:
    print(f"Thresholded values range: [{F_thresh.min():.3f}, {F_thresh.max():.3f}]")
