"""Configuration settings for the dataset generation."""

import os
from pathlib import Path

# Get the root directory (where the git repo is)
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Grid and simulation parameters
N_POINTS = 128
T = -5  # Initial time horizon
MAX_TIME = -40  # Maximum time horizon to try
CONVERGENCE_THRESHOLD = 1e-3

# Environment parameters
DEFAULT_ENV_WIDTH = 10
DEFAULT_ENV_HEIGHT = 10
DEFAULT_NUM_OBSTACLES = 3

# Dataset parameters
NUM_SAMPLES = 600  # Number of examples to generate
GLOBAL_SEED = 42  # Global random seed for reproducibility

# Obstacle parameters
MIN_OBSTACLE_SIZE_RATIO = 0.05
MAX_OBSTACLE_SIZE_RATIO = 0.15

# Output parameters
OUTPUT_DIR = ROOT_DIR / 'dataset' / 'outputs'
SAMPLE_DIR_PREFIX = 'sample_'
ENVIRONMENT_PLOT_NAME = 'environment.png'
ENVIRONMENT_GRID_NAME = 'environment_grid.npy'
VALUE_FUNCTION_NAME = 'value_function.npy'
RESULTS_CSV_NAME = 'results.csv'

# CSV column names
CSV_COLUMNS = [
    'sample_id',
    'seed',
    'converged',
    'convergence_time',
    'final_time_horizon',
    'timestamp'
] 