import numpy as np
import os
from pathlib import Path

def verify_slices_identical(array):
    """Verify that all slices along the last dimension are identical."""
    first_slice = array[..., 0]
    for i in range(1, array.shape[-1]):
        if not np.array_equal(array[..., i], first_slice):
            return False
    return True

def process_directory(directory_path):
    """Process all environment_grid.npy files in the given directory."""
    directory = Path(directory_path)
    
    # Find all environment_grid.npy files
    grid_files = list(directory.glob('**/environment_grid.npy'))
    
    print(f"Found {len(grid_files)} environment grid files to process")
    
    for grid_file in grid_files:
        try:
            # Load the original file
            original_data = np.load(grid_file)
            
            # Check if the array has the expected shape (_, _, 101)
            if len(original_data.shape) != 3 or original_data.shape[-1] != 101:
                print(f"Skipping {grid_file}: Unexpected shape {original_data.shape}")
                continue
            
            # Verify all slices are identical
            if not verify_slices_identical(original_data):
                print(f"Skipping {grid_file}: Slices are not identical")
                continue
            
            # Create reduced version (take first slice)
            reduced_data = original_data[..., 0]
            
            # Rename original file
            original_backup = grid_file.with_name('environment_grid_original.npy')
            os.rename(grid_file, original_backup)
            
            # Save reduced version
            np.save(grid_file, reduced_data)
            
            print(f"Successfully processed {grid_file}")
            
        except Exception as e:
            print(f"Error processing {grid_file}: {str(e)}")

if __name__ == "__main__":
    target_dir = "/Users/malte/AA-276-Diffusion-BRT/dataset/outputs/10_May_2025_05_33"
    process_directory(target_dir) 