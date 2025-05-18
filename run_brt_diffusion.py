import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from dataset.BRT_Dataset import BRT_Dataset
from diffusion.film_module import UNet_conditional_FiLM, EMA, GridProjection
from torch.utils.tensorboard import SummaryWriter
from utils.debug_shapes import debug_shapes
from diffusion.Diffusion import Diffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(args):
    channels = 64
    grid_size = 64

    model = UNet_conditional_FiLM(c_in=channels, c_out=channels, grid_size=grid_size, device=device)
    grid_projection = GridProjection(in_channels=1, out_channels=256, cond_dim=256)#.to(device)

    dataset = BRT_Dataset(args.dataset_path, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=grid_size, noise_steps=1000, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            val_func, grid = batch
            # val_func = val_func#.to(device)
            # grid = grid#.to(device)

            conditioning = grid_projection(grid)
            
            t = diffusion.sample_timesteps(val_func.shape[0])#.to(device)
            x_t, noise = diffusion.noise_images(val_func, t)

            predicted_noise = model(x_t, t, conditioning)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # if epoch % 10 == 0:
        #     sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        #     ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        #     plot_images(sampled_images)
        #     save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #     save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
        #     torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        #     torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        #     torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 2  # Reduced for quick testing
    args.batch_size = 1
    args.dataset_path = "/Users/johncao/Documents/Programming/Stanford/AA276/project/dataset_64"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 3e-4
    
    # Create necessary directories
   # os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
   # os.makedirs(os.path.join("models", args.run_name), exist_ok=True)
    
    train(args)


if __name__ == '__main__':
    launch()
    #debug_shapes()  # Run the debug function instead of training
