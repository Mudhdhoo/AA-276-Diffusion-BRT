import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union


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
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
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


class FiLM_DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, mid_channels=None, residual=False, n_groups=8):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(n_groups, mid_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        
        # FiLM conditioning layer - predicts per-channel scale and bias
        self.cond_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        
        # Make sure dimensions are compatible for residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        self.out_channels = out_channels
        
    def forward(self, x, cond):
        '''
        x: [batch_size, in_channels, height, width]
        cond: [batch_size, cond_dim]
        '''
        # First convolution block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Second convolution block
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Apply FiLM conditioning
        film_params = self.cond_encoder(cond)
        film_params = film_params.view(film_params.size(0), 2, self.out_channels, 1, 1)
        scale = film_params[:, 0]  # [batch, out_channels, 1, 1]
        bias = film_params[:, 1]   # [batch, out_channels, 1, 1]
        
        h = scale * h + bias
        
        # Residual connection
        if self.residual:
            h = F.gelu(h + self.residual_conv(x))
        
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class FiLM_Down(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=256, n_groups=8):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.res_block1 = FiLM_DoubleConv(in_channels, in_channels, cond_dim, residual=True, n_groups=n_groups)
        self.res_block2 = FiLM_DoubleConv(in_channels, out_channels, cond_dim, n_groups=n_groups)

    def forward(self, x, cond):
        x = self.maxpool(x)
        x = self.res_block1(x, cond)
        x = self.res_block2(x, cond)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class FiLM_Up(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=256, n_groups=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block1 = FiLM_DoubleConv(in_channels, in_channels, cond_dim, residual=True, n_groups=n_groups)
        self.res_block2 = FiLM_DoubleConv(in_channels, out_channels, cond_dim, mid_channels=in_channels//2, n_groups=n_groups)

    def forward(self, x, skip_x, cond):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res_block1(x, cond)
        x = self.res_block2(x, cond)
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


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
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

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

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

        if y is not None:
            t += self.label_emb(y)

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


class UNet_FiLM(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, cond_dim=None, n_groups=8, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # Set cond_dim equal to time_dim if not provided
        if cond_dim is None:
            cond_dim = time_dim
        
        # Initial feature extraction without conditioning
        self.inc = DoubleConv(c_in, 64)
        
        # Time embedding
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Combine time embedding with condition
        self.combined_cond_dim = time_dim + cond_dim
        
        # Downsampling path
        self.down1 = FiLM_Down(64, 128, self.combined_cond_dim, n_groups=n_groups)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = FiLM_Down(128, 256, self.combined_cond_dim, n_groups=n_groups)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = FiLM_Down(256, 256, self.combined_cond_dim, n_groups=n_groups)
        self.sa3 = SelfAttention(256, 8)

        # Bottleneck
        self.bot1 = FiLM_DoubleConv(256, 512, self.combined_cond_dim, n_groups=n_groups)
        self.bot2 = FiLM_DoubleConv(512, 512, self.combined_cond_dim, n_groups=n_groups)
        self.bot3 = FiLM_DoubleConv(512, 256, self.combined_cond_dim, n_groups=n_groups)

        # Upsampling path
        self.up1 = FiLM_Up(512, 128, self.combined_cond_dim, n_groups=n_groups)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = FiLM_Up(256, 64, self.combined_cond_dim, n_groups=n_groups)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = FiLM_Up(128, 64, self.combined_cond_dim, n_groups=n_groups)
        self.sa6 = SelfAttention(64, 64)
        
        # Output convolution
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x, t, cond=None):
        """
        x: [B, C, H, W] - Input image
        t: [B] - Diffusion timestep
        cond: [B, cond_dim] - Conditional input (e.g., grid conditions)
        """
        # Encode timestep
        t = t.unsqueeze(-1).type(torch.float)
        time_emb = self.time_encoder(t)
        
        # Combine time embedding with condition
        if cond is None:
            cond = torch.zeros(x.shape[0], self.combined_cond_dim - self.time_dim, device=self.device)
            
        combined_cond = torch.cat([time_emb, cond], dim=-1)
            
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling
        x2 = self.down1(x1, combined_cond)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, combined_cond)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, combined_cond)
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bot1(x4, combined_cond)
        x4 = self.bot2(x4, combined_cond)
        x4 = self.bot3(x4, combined_cond)

        # Upsampling
        x = self.up1(x4, x3, combined_cond)
        x = self.sa4(x)
        x = self.up2(x, x2, combined_cond)
        x = self.sa5(x)
        x = self.up3(x, x1, combined_cond)
        x = self.sa6(x)
        
        # Output
        output = self.outc(x)
        return output


# Sinusoidal position embedding for diffusion step
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


if __name__ == '__main__':
    # net = UNet(device="cpu")
    # net = UNet_conditional(num_classes=10, device="cpu")
    net = UNet_FiLM(c_in=3, c_out=3, time_dim=256, cond_dim=16, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    cond = torch.randn(3, 16)  # Example grid condition encoded as a vector
    print(net(x, t, cond).shape)
