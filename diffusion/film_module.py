import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union
from loguru import logger


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, groups=3):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

# Need to modify this to use FiLM
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=256):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        # FiLM generator
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

    def forward(self, x, cond):
        x = self.maxpool_conv(x)    # (B, C, H, W)

        emb = self.cond_encoder(cond)    # (B, C*2)
        gamma, beta = emb[:, :self.out_channels].unsqueeze(-1).unsqueeze(-1), emb[:, self.out_channels:].unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM
        x = gamma*x + beta

        return x

# Need to modify this to use FiLM
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=256):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # FiLM generator
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

    def forward(self, x, skip_x, cond):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)

        # Apply FiLM
        emb = self.cond_encoder(cond)
        gamma, beta = emb[:, :self.out_channels].unsqueeze(-1).unsqueeze(-1), emb[:, self.out_channels:].unsqueeze(-1).unsqueeze(-1)
        x = gamma*x + beta

        return x

class Mid(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, cond_dim=256):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        if mid_channels:
            self.mid_channels = mid_channels
        else:
            self.mid_channels = in_channels

        self.conv = nn.Sequential(
            DoubleConv(in_channels, mid_channels, residual=True),
            DoubleConv(mid_channels, mid_channels),
            DoubleConv(mid_channels, out_channels),
        )
        
        # FiLM generator
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

    def forward(self, x, cond):
        x = self.conv(x)

        # Apply FiLM
        emb = self.cond_encoder(cond)
        gamma, beta = emb[:, :self.out_channels].unsqueeze(-1).unsqueeze(-1), emb[:, self.out_channels:].unsqueeze(-1).unsqueeze(-1)
        x = gamma*x + beta

        return x


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional_FiLM(nn.Module):
    def __init__(self, c_in=1, c_out=256, grid_size=101, time_dim=256, grid_cond_dim=256,device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.grid_size = grid_size
        self.grid_cond_dim = grid_cond_dim
        cond_dim = self.time_dim + self.grid_cond_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, cond_dim)
        self.sa1 = SelfAttention(128, self.grid_size//2)
        self.down2 = Down(128, 256, cond_dim)
        self.sa2 = SelfAttention(256, self.grid_size//4)
        self.down3 = Down(256, 256, cond_dim)
        self.sa3 = SelfAttention(256, self.grid_size//8)

        self.mid = Mid(256, 256, 256, cond_dim)

        self.up1 = Up(512, 128, cond_dim)
        self.sa4 = SelfAttention(128, self.grid_size//4)
        self.up2 = Up(256, 64, cond_dim)
        self.sa5 = SelfAttention(64, self.grid_size//2)
        self.up3 = Up(128, 64, cond_dim)
        self.sa6 = SelfAttention(64, self.grid_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Concatenate grid condition with time condition
        if y is not None:
            t = torch.cat([t, y], dim=1)
            
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.mid(x4, t)
       # x4 = self.bot1(x4)
        #x4 = self.bot2(x4)
       # x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output


class GridProjection(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        
        self.conv_block1 = nn.Sequential(   
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(   
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )   

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.MaxPool2d(2)
        )   

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )   

        self.pool = nn.AdaptiveAvgPool2d((1,1))

       # proj_dim = out_channels * 
        self.proj = nn.Linear(out_channels, cond_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x) 
        x = self.conv_block3(x) + x
        x = self.conv_block4(x)
        x = self.conv_block5(x) + x
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)

        return x
    

if __name__ == "__main__":
    x = torch.randn(2, 1, 64, 64)
    y = torch.randn(2, 101, 101)
    model = GridProjection(1, 256)
    print(model(x, y).shape)
