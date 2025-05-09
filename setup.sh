#!/bin/bash

# Function to check if NVIDIA GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi &> /dev/null
        return $?
    else
        return 1
    fi
}

# Function to install CPU dependencies
install_cpu() {
    echo "Installing CPU dependencies..."
    pip install jax jaxlib numpy matplotlib scipy hj_reachability scikit-image tqdm
}

# Function to install GPU dependencies
install_gpu() {
    echo "Installing GPU dependencies..."
    # Install base packages
    pip install numpy matplotlib scipy hj_reachability scikit-image tqdm
    
    # Install JAX with CUDA support
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # Install CUDA toolkit if not present
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA toolkit not found. Please install CUDA toolkit first."
        echo "You can install it using:"
        echo "sudo apt-get update && sudo apt-get install -y cuda-toolkit"
        exit 1
    fi
}

# Main script
echo "Setting up environment for AA-276-Diffusion-BRT..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Check if running in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not running in a virtual environment."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for GPU
if check_gpu; then
    echo "NVIDIA GPU detected!"
    read -p "Do you want to use GPU? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_gpu
        echo "GPU setup complete!"
    else
        install_cpu
        echo "CPU setup complete!"
    fi
else
    echo "No NVIDIA GPU detected. Installing CPU version..."
    install_cpu
    echo "CPU setup complete!"
fi

# Verify installation
echo "Verifying installation..."
python3 -c "import jax; print('JAX version:', jax.__version__)"
if check_gpu; then
    python3 -c "import jax; print('JAX devices:', jax.devices())"
fi

echo "Setup complete!" 