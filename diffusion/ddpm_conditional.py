import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_FiLM, EMA
import logging
import wandb
from torch.utils.data import DataLoader
from dataset.value_function_dataset import create_value_function_dataset
from dataset.config import N_POINTS, OUTPUT_DIR, RESULTS_CSV_NAME

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.run_name)
    
    # Create dataset and dataloader
    dataset = create_value_function_dataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model with correct dimensions
    model = UNet_FiLM(
        c_in=N_POINTS,  # Value function has N_POINTS channels
        c_out=N_POINTS,  # Value function has N_POINTS channels
        time_dim=256,
        grid_size=N_POINTS,
        cond_dim=args.cond_dim,
        device=device
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=N_POINTS, device=device)  # Using grid size from config
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            # Get environment grid and value function
            env_grid = batch['environment_grid'].to(device)  # [B, 1, N_POINTS, N_POINTS]
            value_function = batch['value_function'].to(device)  # [B, N_POINTS, N_POINTS, N_POINTS]
            
            # Add channel dimension to env_grid if needed
            if len(env_grid.shape) == 3:
                env_grid = env_grid.unsqueeze(1)
            
            t = diffusion.sample_timesteps(env_grid.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(value_function, t)
            
            # Use environment grid as condition
            predicted_noise = model(x_t, t, env_grid)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            
            # Log to wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/step": epoch * l + i
            })

        if epoch % 10 == 0:
            # Sample with a few environment grids
            sample_env_grids = env_grid[:4]  # Take first 4 grids
            sampled_value_functions = diffusion.sample(model, n=len(sample_env_grids), labels=sample_env_grids)
            ema_sampled_value_functions = diffusion.sample(ema_model, n=len(sample_env_grids), labels=sample_env_grids)
            
            # Log samples to wandb
            wandb.log({
                "samples/epoch": epoch,
                "samples/value_functions": wandb.Image(sampled_value_functions),
                "samples/ema_value_functions": wandb.Image(ema_sampled_value_functions)
            })
            
            # Save checkpoints
            torch.save(model.state_dict(), os.path.join(args.output_dir, "models", f"{args.run_name}_ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.output_dir, "models", f"{args.run_name}_ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "models", f"{args.run_name}_optim.pt"))
    
    wandb.finish()


def launch():
    import argparse
    import getpass
    import os
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Interactive setup
    print("\n=== Value Function Diffusion Training Setup ===\n")
    
    # Create output directories
    args.output_dir = os.path.join("diffusion", "outputs")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Wandb setup
    print("Setting up Weights & Biases logging...")
    try:
        import wandb
        if not wandb.api.api_key:
            print("No wandb API key found. Please enter your API key:")
            wandb_api_key = getpass.getpass("Wandb API key: ")
            wandb.login(key=wandb_api_key)
    except ImportError:
        print("wandb not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "wandb"])
        import wandb
        print("Please enter your wandb API key:")
        wandb_api_key = getpass.getpass("Wandb API key: ")
        wandb.login(key=wandb_api_key)
    
    # Project setup
    args.wandb_project = input("Enter wandb project name [value-function-diffusion]: ") or "value-function-diffusion"
    args.run_name = input("Enter run name [DDPM_ValueFunction]: ") or "DDPM_ValueFunction"
    
    # Dataset setup
    default_dataset_path = os.path.join(OUTPUT_DIR, RESULTS_CSV_NAME)
    args.dataset_path = input(f"Enter path to results.csv [{default_dataset_path}]: ") or default_dataset_path
    
    # Training parameters
    try:
        args.epochs = int(input("Enter number of epochs [300]: ") or "300")
    except ValueError:
        print("Invalid input, using default value of 300")
        args.epochs = 300
        
    try:
        args.batch_size = int(input("Enter batch size [32]: ") or "32")
    except ValueError:
        print("Invalid input, using default value of 32")
        args.batch_size = 32
        
    try:
        args.lr = float(input("Enter learning rate [3e-4]: ") or "3e-4")
    except ValueError:
        print("Invalid input, using default value of 3e-4")
        args.lr = 3e-4
    
    # Fixed parameters
    args.image_size = N_POINTS  # Grid size from config
    args.cond_dim = 256    # Fixed condition dimension
    
    # Device setup
    if torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"
    args.device = input(f"Enter device [{default_device}]: ") or default_device
    
    # Print configuration
    print("\n=== Training Configuration ===")
    print(f"Project: {args.wandb_project}")
    print(f"Run name: {args.run_name}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grid size: {args.image_size} (fixed)")
    print(f"Condition dimension: {args.cond_dim} (fixed)")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Confirm before starting
    confirm = input("\nStart training with these settings? [y/N]: ").lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\nStarting training...")
    train(args)


if __name__ == '__main__':
    launch()
