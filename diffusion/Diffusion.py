import torch
import logging
from tqdm import tqdm
from loguru import logger
from dataset.statistics.stats import MEAN_VAL_FUNC, STD_VAL_FUNC

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
        
        # Move normalization tensors to the specified device
        self.mean_val_func = torch.tensor(MEAN_VAL_FUNC).to(device)
        self.std_val_func = torch.tensor(STD_VAL_FUNC).to(device)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, grid_projection, n, dataset, cfg_scale=0):
        logging.info(f"Sampling {n} new images....")
        env_grids = dataset.sample_env_grid(n)
        env_grids = env_grids.to(self.device)
        env_grids = grid_projection(env_grids)
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 64, self.img_size, self.img_size)).to(self.device)              # dont hardcode 64 change later
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, env_grids)

                # if cfg_scale > 0:
                #     uncond_predicted_noise = model(x, t, None)
                #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()

        x = x*self.std_val_func + self.mean_val_func

        return x