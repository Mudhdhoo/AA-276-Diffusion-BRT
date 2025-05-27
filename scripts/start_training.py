import subprocess
import os
import time
import signal
import sys
import wandb
import argparse

def run_training_process(run_name, dataset_dir, beta_end, cfg_scale, batch_size, wandb_api_key):
    """Start a training process with the given configuration"""
    cmd = [
        "python", "diffusion_pointcloud.py",
        "--dataset_dir", dataset_dir,
        "--beta_end", str(beta_end),
        "--guidance_scale", str(cfg_scale),
        "--batch_size", str(batch_size),
        "--wandb_api_key", wandb_api_key,
        "--wandb_project", "brt-diffusion"
    ]
    return subprocess.Popen(cmd)

def main():
    parser = argparse.ArgumentParser(description='Start multiple training runs with different configurations')
    parser.add_argument('--wandb_api_key', type=str, required=True,
                      help='Weights & Biases API key')
    args = parser.parse_args()

    # Start all training processes
    processes = []
    
    # Run A
    processes.append(run_training_process(
        run_name="run_a_cfg015",
        dataset_dir="~/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv",
        beta_end=0.008,
        cfg_scale=0.15,
        batch_size=128,
        wandb_api_key=args.wandb_api_key
    ))
    
    # Run B
    processes.append(run_training_process(
        run_name="run_b_cfg020",
        dataset_dir="~/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv",
        beta_end=0.008,
        cfg_scale=0.2,
        batch_size=128,
        wandb_api_key=args.wandb_api_key
    ))
    
    # Run C
    processes.append(run_training_process(
        run_name="run_c_cfg030",
        dataset_dir="~/1070_4d_pointcloud_3000inside_1000outside_4cloudsperenv",
        beta_end=0.008,
        cfg_scale=0.3,
        batch_size=128,
        wandb_api_key=args.wandb_api_key
    ))

    try:
        # Wait for all processes to complete
        for p in processes:
            p.wait()
        
        print("All training processes have completed.")
        
        # Shutdown the machine
        print("Shutting down the machine...")
        os.system("sudo shutdown -h now")
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Terminating all processes...")
        for p in processes:
            p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
