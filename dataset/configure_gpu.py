"""GPU configuration script for the dataset generation.

This module handles the setup and verification of GPU resources for the dataset generation.
It configures memory allocation, CUDA device selection, and verifies the JAX installation.
"""

import os
import sys
import subprocess
import platform
from .config import GPU_MEMORY_FRACTION, CUDA_VISIBLE_DEVICES

def check_cuda_installation():
    """Check if CUDA is properly installed and accessible.
    
    Returns:
        bool: True if CUDA is properly installed and accessible, False otherwise.
    """
    try:
        # Try to run nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("CUDA is properly installed:")
            print(result.stdout)
            return True
        else:
            print("Error running nvidia-smi:", result.stderr)
            return False
    except FileNotFoundError:
        print("nvidia-smi not found. CUDA might not be installed.")
        return False

def configure_gpu_environment():
    """Configure GPU environment variables and settings.
    
    This function sets up the GPU environment by configuring memory allocation
    and device selection using the values from config.py.
    """
    # Set GPU memory management
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(GPU_MEMORY_FRACTION)
    
    # Set CUDA visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    
    # Print configuration
    print("\nGPU Environment Configuration:")
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    print(f"XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def verify_python_packages():
    """Verify that required GPU packages are installed and properly configured.
    
    Returns:
        bool: True if all required packages are installed and configured correctly,
              False otherwise.
    """
    try:
        import jax
        import jaxlib
        print("\nJAX Configuration:")
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX default backend: {jax.default_backend()}")
        
        # Check if CUDA is available
        if jax.default_backend() == 'gpu':
            print("JAX is configured to use GPU")
        else:
            print("JAX is configured to use CPU")
            
    except ImportError as e:
        print(f"Error importing required packages: {e}")
        return False
    
    return True

def main():
    """Main function to configure and verify GPU setup.
    
    This function orchestrates the GPU configuration process by:
    1. Checking CUDA installation
    2. Configuring GPU environment variables
    3. Verifying Python package installation
    
    Returns:
        bool: True if configuration was successful, False otherwise.
    """
    print("Starting GPU configuration check...")
    
    # Check CUDA installation
    has_gpu = check_cuda_installation()
    
    if has_gpu:
        # Configure GPU environment
        configure_gpu_environment()
    else:
        print("No GPU detected, using CPU")
    
    # Verify Python packages
    if not verify_python_packages():
        print("Python package verification failed. Please check your installation.")
        return False
    
    print("\nConfiguration completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 