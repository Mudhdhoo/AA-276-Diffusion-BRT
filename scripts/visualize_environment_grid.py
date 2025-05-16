import numpy as np
import matplotlib.pyplot as plt

# Load the environment grid
data = np.load('/Users/malte/AA-276-Diffusion-BRT/dataset/outputs/16_May_2025_12_23/sample_000/environment_grid.npy')

# Create coordinate vectors
x = np.linspace(0, 10, data.shape[0])  # Assuming 10x10 environment
y = np.linspace(0, 10, data.shape[1])

# Create a figure with a single subplot
plt.figure(figsize=(10, 8))

# Use pcolormesh with explicit coordinates
plt.pcolormesh(x, y, data.T, cmap='binary', shading='auto')
plt.colorbar(label='Obstacle (1) / Free Space (0)')
plt.title('Environment Grid Visualization')
plt.xlabel('X')
plt.ylabel('Y')

# Set equal aspect ratio
plt.axis('equal')

# Save the plot
plt.savefig('environment_grid_visualization.png')
plt.close()

# Print array information
print(f"Array shape: {data.shape}")
print(f"Array dtype: {data.dtype}")
print(f"Min value: {data.min()}")
print(f"Max value: {data.max()}")
print(f"Mean value: {data.mean()}")
print(f"Number of obstacle cells (1.0): {np.sum(data == 1.0)}")
print(f"Number of free space cells (0.0): {np.sum(data == 0.0)}") 