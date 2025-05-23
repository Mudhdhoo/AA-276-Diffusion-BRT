import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataset.BRT_Dataset import BRT_Dataset
from diffusion.diff_3d.film_module_3d import UNet_conditional_FiLM_3D, EMA, GridProjection3D
from torch.utils.tensorboard import SummaryWriter
from diffusion.diff_3d.Diffusion3D import Diffusion3D
from diffusion.diff_3d.utils_3d import plot_value_function_batch_3d
from loguru import logger
from torch.cuda.amp import autocast, GradScaler


device = "cuda" if torch.cuda.is_available() else "cpu"

def train(args):
    channels = 1  # For 3D, typically 1 channel (value function)
    grid_size = 64  # Adjust as needed for your 3D grid

    model = UNet_conditional_FiLM_3D(c_in=channels, c_out=channels, grid_size=grid_size, device=device).to(device)
    grid_projection = GridProjection3D(in_channels=1, out_channels=256, cond_dim=256).to(device)

    dataset = BRT_Dataset(args.dataset_path, split="train")
    dataset_test = BRT_Dataset(args.dataset_path, split="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    mse = nn.MSELoss()
    scaler = GradScaler()  # For mixed precision training

    diffusion = Diffusion3D(img_size=grid_size, 
                          noise_steps=args.noise_steps, 
                          beta_start=args.beta_start, 
                          beta_end=args.beta_end, 
                          device=device)
    summary_writer = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            val_func, grid = batch
            val_func = val_func.to(device) # Shape should be (B, 1, D, H, W)
            grid = grid.unsqueeze(2).to(device)          # (B, 1, 1, H, W) because GridProjection expects this

            conditioning = grid_projection(grid)
            t = diffusion.sample_timesteps(val_func.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(val_func, t)

            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                predicted_noise = model(x_t, t, conditioning)
                loss = mse(noise, predicted_noise)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            summary_writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % args.save_every == 0:
            sampled_images = diffusion.sample(model, grid_projection, args.num_samples_plot, dataset_test)
            ema_sampled_images = diffusion.sample(ema_model, grid_projection, args.num_samples_plot, dataset_test)
            plot_value_function_batch_3d(sampled_images, time=None, 
                                      save_path=os.path.join("results", args.run_name, f"{epoch}.png"))
            plot_value_function_batch_3d(ema_sampled_images, time=None, 
                                      save_path=os.path.join("results", args.run_name, f"{epoch}_ema.png"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            torch.save(scheduler.state_dict(), os.path.join("models", args.run_name, f"scheduler.pt"))
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        summary_writer.add_scalar("Learning_Rate", current_lr, global_step=epoch)
        logger.info(f"Epoch {epoch} completed. Learning rate: {current_lr}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="BRT_diffusion_3D")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_path", type=str, default="dataset_64")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_step_size", type=int, default=600, help="Number of epochs between learning rate decay")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Multiplicative factor of learning rate decay")
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--num_samples_plot", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()

def main():
    args = get_args()    
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    os.makedirs(os.path.join("models", args.run_name), exist_ok=True)
    train(args)

if __name__ == '__main__':
    main() 