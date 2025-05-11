import numpy as np
from pathlib import Path

def verify_and_cleanup(directory_path):
    """Verify reduced files and clean up original files."""
    directory = Path(directory_path)
    
    # Find all reduced environment grid files
    reduced_files = list(directory.glob('**/environment_grid.npy'))
    original_files = list(directory.glob('**/environment_grid_original.npy'))
    
    print(f"Found {len(reduced_files)} reduced files and {len(original_files)} original files")
    
    if len(reduced_files) != 1200:
        print(f"WARNING: Expected 1200 reduced files, found {len(reduced_files)}")
        return
    
    if len(original_files) != 1200:
        print(f"WARNING: Expected 1200 original files, found {len(original_files)}")
        return
    
    # Verify all reduced files are 2D
    for reduced_file in reduced_files:
        try:
            data = np.load(reduced_file)
            if len(data.shape) != 2:
                print(f"WARNING: {reduced_file} is not 2D, shape is {data.shape}")
                return
        except Exception as e:
            print(f"Error loading {reduced_file}: {str(e)}")
            return
    
    print("All verifications passed. Proceeding with cleanup...")
    
    # Delete original files
    for original_file in original_files:
        try:
            original_file.unlink()
            print(f"Deleted {original_file}")
        except Exception as e:
            print(f"Error deleting {original_file}: {str(e)}")
    
    print("Cleanup complete!")

if __name__ == "__main__":
    target_dir = "/Users/malte/AA-276-Diffusion-BRT/dataset/outputs/10_May_2025_05_33"
    verify_and_cleanup(target_dir) 