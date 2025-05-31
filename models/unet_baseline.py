import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net - optimized for simple environments"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    """U-Net based encoder for environment grids - optimized for simple rectangle environments"""
    def __init__(self, env_size, output_dim=128, dropout_rate=0.2):
        super().__init__()
        self.env_size = env_size
        
        # Verify env_size is compatible with downsampling
        assert env_size % 16 == 0, f"env_size ({env_size}) must be divisible by 16 for 4 downsampling layers"
        
        # Reduced channel progression for simple environments (rectangles)
        # Original: 64->128->256->512->1024
        # Optimized: 32->64->128->256->256 (much smaller for simple shapes)
        self.inc = DoubleConv(1, 32, dropout_rate)
        self.down1 = Down(32, 64, dropout_rate)
        self.down2 = Down(64, 128, dropout_rate)
        self.down3 = Down(128, 256, dropout_rate)
        self.down4 = Down(256, 256, dropout_rate)  # Keep same size to reduce params
        
        # Global feature extraction from bottleneck
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Reduced FC layers for simple environments
        self.fc1 = nn.Linear(256, 256)  # Reduced from 1024->512
        self.fc2 = nn.Linear(256, 128)  # Reduced from 512->256
        self.fc3 = nn.Linear(128, output_dim)  # Keep output_dim flexible
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, env):
        # env shape: (batch_size, env_size, env_size)
        x = env.unsqueeze(1)  # Add channel dimension
        
        # U-Net encoder path
        x1 = self.inc(x)      # 32 channels
        x2 = self.down1(x1)   # 64 channels, /2
        x3 = self.down2(x2)   # 128 channels, /4
        x4 = self.down3(x3)   # 256 channels, /8
        x5 = self.down4(x4)   # 256 channels, /16
        
        # Global feature extraction
        global_features = self.global_pool(x5)  # (B, 256, 1, 1)
        global_features = global_features.view(global_features.size(0), -1)  # (B, 256)
        
        # Project to desired output dimension
        x = self.activation(self.fc1(global_features))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class PointGenerator(nn.Module):
    """MLP that generates point cloud from environment encoding - balanced for expressiveness"""
    def __init__(self, env_encoding_dim, num_points, state_dim, 
                 hidden_dims=None, dropout_rate=0.3):
        super().__init__()
        self.num_points = num_points
        self.state_dim = state_dim
        self.output_dim = num_points * state_dim
        
        # Balanced hidden dimensions for 4000 points
        # Still deep enough for complex point clouds but much more compact
        if hidden_dims is None:
            # Adaptive sizing based on output requirements
            if num_points <= 1000:
                hidden_dims = [256, 512, 256]
            elif num_points <= 2000:
                hidden_dims = [256, 512, 512, 256]
            else:  # 4000+ points - need more capacity
                hidden_dims = [512, 1024, 512, 256]
        
        # Verify dimension compatibility
        assert len(hidden_dims) > 0, "hidden_dims must not be empty"
        assert env_encoding_dim > 0, "env_encoding_dim must be positive"
        
        # Build MLP with residual connections where possible
        self.input_proj = nn.Linear(env_encoding_dim, hidden_dims[0])
        
        # Main processing blocks with residual connections where possible
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate * (1 + i * 0.1))  # Progressive dropout
            )
            self.blocks.append(block)
        
        # Final output layer
        self.output_proj = nn.Linear(hidden_dims[-1], self.output_dim)
    
    def forward(self, env_encoding):
        # env_encoding shape: (batch_size, env_encoding_dim)
        x = self.input_proj(env_encoding)
        
        # Pass through blocks with residual connections where dimensions match
        prev_x = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Add residual connection if dimensions match
            if x.shape == prev_x.shape:
                x = x + prev_x
            prev_x = x
        
        # Final projection
        output = self.output_proj(x)
        
        # Reshape to point cloud format
        batch_size = output.shape[0]
        points = output.view(batch_size, self.num_points, self.state_dim)
        
        return points


class BRTUNet(nn.Module):
    """U-Net based model for BRT generation - optimized for simple environments with 4000 points"""
    def __init__(self, env_size, num_points, max_state_dim=4, env_encoding_dim=128, 
                 dropout_rate=0.2, weight_decay_strength=0.01):
        super().__init__()
        self.env_size = env_size
        self.num_points = num_points
        self.max_state_dim = max_state_dim
        self.weight_decay_strength = weight_decay_strength
        
        # U-Net encoder for environment processing
        self.env_encoder = UNetEncoder(
            env_size, 
            output_dim=env_encoding_dim,
            dropout_rate=dropout_rate
        )
        
        # Point generator with adaptive sizing
        self.point_generator = PointGenerator(
            env_encoding_dim, 
            num_points, 
            max_state_dim,
            dropout_rate=dropout_rate
        )
        
        # Apply weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_l2_regularization(self):
        """Compute L2 regularization term"""
        l2_reg = 0
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2) ** 2
        return self.weight_decay_strength * l2_reg
    
    def forward(self, env, target_state_dim=None):
        """
        Forward pass
        Args:
            env: (batch_size, env_size, env_size) - environment grid
            target_state_dim: desired output dimension (3 or 4)
        Returns:
            points: (batch_size, num_points, state_dim) - generated BRT point cloud
        """
        # Encode environment using U-Net
        env_encoding = self.env_encoder(env)
        
        # Generate points
        points = self.point_generator(env_encoding)
        
        # Slice to desired dimensions if needed
        if target_state_dim is not None and target_state_dim < self.max_state_dim:
            points = points[:, :, :target_state_dim]
        
        return points
    
    def compute_chamfer_loss(self, pred_points, target_points):
        """Compute permutation-invariant Chamfer distance loss"""
        # Verify tensor shapes (must be rank-3 tensors: batch_size x num_points x feature_dim)
        assert len(pred_points.shape) == 3, f"pred_points must be rank-3 tensor (B,N,D), got shape {pred_points.shape}"
        assert len(target_points.shape) == 3, f"target_points must be rank-3 tensor (B,M,D), got shape {target_points.shape}"
        
        B, N, D = pred_points.shape
        B_target, M, D_target = target_points.shape
        
        # Ensure batch sizes match
        assert B == B_target, f"Batch size mismatch: pred_points has {B}, target_points has {B_target}"
        
        # Ensure feature dimensions match
        assert D == D_target, f"Feature dimension mismatch: pred_points has {D}, target_points has {D_target}"
        
        # Expand dimensions for pairwise distance computation
        pred_expanded = pred_points.unsqueeze(2)    # (B, N, 1, D)
        target_expanded = target_points.unsqueeze(1)  # (B, 1, M, D)
        
        # Compute pairwise squared distances
        dists = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)  # (B, N, M)
        
        # Chamfer distance: average of nearest neighbor distances in both directions
        dist_pred_to_target = torch.min(dists, dim=2)[0].mean(dim=1)  # (B,)
        dist_target_to_pred = torch.min(dists, dim=1)[0].mean(dim=1)  # (B,)
        
        chamfer_loss = (dist_pred_to_target + dist_target_to_pred).mean()
        
        return chamfer_loss    
    
    def compute_loss(self, pred_points, target_points, include_l2_reg=True):
        """
        Compute simple Chamfer distance loss with optional L2 regularization
        This is simple and requires no hyperparameter tuning
        """
        # Ensure dimensions match
        min_dim = min(pred_points.shape[-1], target_points.shape[-1])
        pred_points = pred_points[:, :, :min_dim]
        target_points = target_points[:, :, :min_dim]
        
        # Compute Chamfer loss (simple and effective)
        chamfer_loss = self.compute_chamfer_loss(pred_points, target_points)
        
        # Add L2 regularization during training
        if include_l2_reg and self.training:
            l2_reg = self.get_l2_regularization()
            total_loss = chamfer_loss + l2_reg
        else:
            total_loss = chamfer_loss
            
        return total_loss
