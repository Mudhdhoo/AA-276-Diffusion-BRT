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
    
    def compute_chamfer_loss(self, pred_points, target_points, chunk_size=500):
            """
            Memory-optimized Chamfer distance computation using chunking
            
            CHANGES MADE:
            1. Added chunk_size parameter to process points in smaller batches
            2. Chunked computation to avoid storing full (B, N, M) distance matrix
            3. Added memory cleanup between chunks
            4. Process each batch item separately if needed for very large point clouds
            """
            # Verify tensor shapes (must be rank-3 tensors: batch_size x num_points x feature_dim)
            assert len(pred_points.shape) == 3, f"pred_points must be rank-3 tensor (B,N,D), got shape {pred_points.shape}"
            assert len(target_points.shape) == 3, f"target_points must be rank-3 tensor (B,M,D), got shape {target_points.shape}"
            
            B, N, D = pred_points.shape
            B_target, M, D_target = target_points.shape
            
            # Ensure batch sizes match
            assert B == B_target, f"Batch size mismatch: pred_points has {B}, target_points has {B_target}"
            
            # Ensure feature dimensions match
            assert D == D_target, f"Feature dimension mismatch: pred_points has {D}, target_points has {D_target}"
            
            # If point clouds are small enough, use original implementation
            if N * M < 1000000:  # Threshold for switching to chunked computation
                return self._compute_chamfer_loss_original(pred_points, target_points)
            
            # For large point clouds, use chunked computation
            total_chamfer_loss = 0.0
            
            for b in range(B):
                pred_batch = pred_points[b:b+1]  # (1, N, D)
                target_batch = target_points[b:b+1]  # (1, M, D)
                
                # Compute distances from pred to target (chunked)
                min_dists_pred_to_target = []
                for i in range(0, N, chunk_size):
                    end_i = min(i + chunk_size, N)
                    pred_chunk = pred_batch[:, i:end_i]  # (1, chunk_size, D)
                    
                    # Expand for distance computation
                    pred_expanded = pred_chunk.unsqueeze(2)  # (1, chunk_size, 1, D)
                    target_expanded = target_batch.unsqueeze(1)  # (1, 1, M, D)
                    
                    # Compute distances for this chunk
                    chunk_dists = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)  # (1, chunk_size, M)
                    min_dists_chunk = torch.min(chunk_dists, dim=2)[0]  # (1, chunk_size)
                    min_dists_pred_to_target.append(min_dists_chunk)
                    
                    # Clear intermediate tensors
                    del pred_expanded, target_expanded, chunk_dists, min_dists_chunk
                
                # Concatenate and compute mean
                min_dists_pred_to_target = torch.cat(min_dists_pred_to_target, dim=1)  # (1, N)
                dist_pred_to_target = min_dists_pred_to_target.mean()
                
                # Compute distances from target to pred (chunked)
                min_dists_target_to_pred = []
                for j in range(0, M, chunk_size):
                    end_j = min(j + chunk_size, M)
                    target_chunk = target_batch[:, j:end_j]  # (1, chunk_size, D)
                    
                    # Expand for distance computation
                    target_expanded = target_chunk.unsqueeze(2)  # (1, chunk_size, 1, D)
                    pred_expanded = pred_batch.unsqueeze(1)  # (1, 1, N, D)
                    
                    # Compute distances for this chunk
                    chunk_dists = torch.sum((target_expanded - pred_expanded) ** 2, dim=-1)  # (1, chunk_size, N)
                    min_dists_chunk = torch.min(chunk_dists, dim=2)[0]  # (1, chunk_size)
                    min_dists_target_to_pred.append(min_dists_chunk)
                    
                    # Clear intermediate tensors
                    del target_expanded, pred_expanded, chunk_dists, min_dists_chunk
                
                # Concatenate and compute mean
                min_dists_target_to_pred = torch.cat(min_dists_target_to_pred, dim=1)  # (1, M)
                dist_target_to_pred = min_dists_target_to_pred.mean()
                
                # Add to total loss
                batch_chamfer_loss = dist_pred_to_target + dist_target_to_pred
                total_chamfer_loss += batch_chamfer_loss
                
                # Clear batch tensors
                del min_dists_pred_to_target, min_dists_target_to_pred
            
            # Return average over batch
            return total_chamfer_loss / B

    def _compute_chamfer_loss_original(self, pred_points, target_points):
        """Original implementation for smaller point clouds"""
        B, N, D = pred_points.shape
        _, M, _ = target_points.shape
        
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

    def compute_emd_loss_approx(self, pred_points, target_points, max_points=800):
        """
        Approximate Earth Mover's Distance using random sampling
        Memory-efficient for large point clouds (4000 points)
        """
        batch_size = pred_points.shape[0]
        num_points = pred_points.shape[1]
        
        total_emd = 0.0
        
        for b in range(batch_size):
            pred_b = pred_points[b]  # (N, D)
            target_b = target_points[b]  # (N, D)
            
            # Sample subset for EMD computation to keep it tractable
            if num_points > max_points:
                idx = torch.randperm(num_points, device=pred_points.device)[:max_points]
                pred_b = pred_b[idx]
                target_b = target_b[idx]
                sample_size = max_points
            else:
                sample_size = num_points
            
            # Compute pairwise distance matrix
            dist_matrix = torch.cdist(pred_b, target_b, p=2)  # (sample_size, sample_size)
            
            # Approximate EMD using Sinkhorn iterations (differentiable approximation)
            # This is much faster than Hungarian algorithm and still effective
            emd_approx = self._sinkhorn_emd(dist_matrix, num_iters=10, reg=0.1)
            total_emd += emd_approx
            
        return total_emd / batch_size
    
    def _sinkhorn_emd(self, cost_matrix, num_iters=10, reg=0.1):
        """
        Sinkhorn approximation of EMD - differentiable and fast
        Based on "Computational Optimal Transport" (Peyré & Cuturi, 2019)
        """
        n = cost_matrix.shape[0]
        
        # Initialize uniform distributions
        a = torch.ones(n, device=cost_matrix.device) / n  # source distribution
        b = torch.ones(n, device=cost_matrix.device) / n  # target distribution
        
        # Sinkhorn iterations
        K = torch.exp(-cost_matrix / reg)  # Gibbs kernel
        u = torch.ones_like(a)
        
        for _ in range(num_iters):
            v = b / (K.T @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        # Compute transport plan
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # EMD is the sum of cost * transport plan
        emd = torch.sum(P * cost_matrix)
        
        return emd

    def compute_density_regularization(self, pred_points, target_points, k=8):
        """
        Density-based regularization to encourage proper point distribution
        Uses k-nearest neighbor distances as density proxy
        """
        batch_size = pred_points.shape[0]
        total_density_loss = 0.0
        
        for b in range(batch_size):
            pred_b = pred_points[b]  # (N, D)
            target_b = target_points[b]  # (N, D)
            
            # Compute k-nearest neighbor distances (density estimation)
            pred_dists = torch.cdist(pred_b, pred_b, p=2)  # (N, N)
            target_dists = torch.cdist(target_b, target_b, p=2)  # (N, N)
            
            # Get k-th nearest neighbor distance (excluding self, so k+1)
            pred_knn = torch.topk(pred_dists, k+1, dim=1, largest=False)[0][:, -1]  # k-th distance
            target_knn = torch.topk(target_dists, k+1, dim=1, largest=False)[0][:, -1]
            
            # Match density distributions
            density_loss = torch.nn.functional.mse_loss(pred_knn, target_knn)
            total_density_loss += density_loss
            
        return total_density_loss / batch_size
        
    def compute_loss(self, pred_points, target_points, include_l2_reg=True):
        """
        Improved loss function with EMD + density regularization
        
        Loss components:
        - 20% Chamfer distance (for basic point matching)
        - 60% EMD (for proper structure and distribution)
        - 20% Density regularization (for local point distribution)
        """
        # Ensure dimensions match
        min_dim = min(pred_points.shape[-1], target_points.shape[-1])
        pred_points = pred_points[:, :, :min_dim]
        target_points = target_points[:, :, :min_dim]
        
        # Component weights (tuned for BRT generation)
        chamfer_weight = 0.2
        emd_weight = 0.6
        density_weight = 0.2
        
        # Compute loss components
        chamfer_loss = self.compute_chamfer_loss(pred_points, target_points)
        emd_loss = self.compute_emd_loss_approx(pred_points, target_points)
        density_loss = self.compute_density_regularization(pred_points, target_points)
        
        # Combine losses
        total_loss = (chamfer_weight * chamfer_loss + 
                     emd_weight * emd_loss + 
                     density_weight * density_loss)
        
        # Add L2 regularization during training
        if include_l2_reg and self.training:
            l2_reg = self.get_l2_regularization()
            total_loss = total_loss + l2_reg
            
        return total_loss