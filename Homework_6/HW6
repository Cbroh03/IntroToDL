import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchinfo import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import SwinForImageClassification

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CPU Fallback option for device configuration.
if torch.cuda.is_available():
    # Device configuration
    device = torch.device('cuda')
    print(f"CUDA is available. Using device: {device}")
else:
    # Device configuration
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

def get_cifar100_loaders(batch_size=64, num_workers=2):
    # Data transforms
    # For ResNet and custom ViT, we'll use standard transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-100 datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

def get_cifar100_swin_loaders(batch_size=32, num_workers=2, img_size=224):
    # Since Swin models are pretrained on ImageNet with 224x224 images,
    # we need to resize CIFAR-100 images to 224x224
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-100 datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

class ResNet18Model(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18Model, self).__init__()
        self.model = resnet18(pretrained=True)
        # Change the first conv layer to accept 32x32 images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool as it's not needed for small images
        # Adjust final fc layer for CIFAR-100 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_resnet18():
    print("Training ResNet-18 baseline on CIFAR-100")

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Load data
    train_loader, test_loader = get_cifar100_loaders(batch_size)

    # Initialize model
    model = ResNet18Model().to(device)

    # Calculate parameters and FLOPs
    model_stats = summary(model, input_size=(batch_size, 3, 32, 32), verbose=0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"FLOPs per forward pass: {model_stats.total_mult_adds:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_start_time = time.time()
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })

        # Measure epoch time
        epoch_time = time.time() - epoch_start_time

        # Test accuracy
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Test Acc: {test_acc:.2f}%")

    # Calculate total training time
    total_training_time = time.time() - total_start_time

    # Save results
    results = {
        'model': 'ResNet-18',
        'total_params': total_params,
        'flops': model_stats.total_mult_adds,
        'total_training_time': total_training_time,
        'avg_epoch_time': total_training_time / num_epochs,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': test_accuracies[-1]
    }

    return results

class PatchEmbedding(nn.Module):
    """
    Split the image into patches and linearly embed them
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Create projection layer
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention module
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Define Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input: (batch_size, n_patches+1, embed_dim)
        batch_size, n_tokens, embed_dim = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, n_tokens, head_dim)

        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, n_tokens, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to V
        x = attn @ v

        # Reshape and project
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, n_tokens, embed_dim)

        x = self.proj(x)
        x = self.dropout(x)

        return x

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Vision Transformer
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    """
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        # Layer Normalization before attention
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        # Layer Normalization before MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )

    def forward(self, x):
        # Apply pre-norm for attention
        attn_output = self.attn(self.norm1(x))
        # Residual connection
        x = x + attn_output
        # Apply pre-norm for MLP
        mlp_output = self.mlp(self.norm2(x))
        # Residual connection
        x = x + mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=100,
        embed_dim=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Number of patches
        self.n_patches = self.patch_embed.n_patches

        # Add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim)
        )

        # Dropout after embedding
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize positional embeddings and class token
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Apply general weight initialization to all linear layers
        self.apply(self._init_weights_general)

    def _init_weights_general(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Prepend class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, n_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed  # (batch_size, n_patches+1, embed_dim)
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Layer normalization
        x = self.norm(x)

        # Take the class token representation
        x = x[:, 0]  # (batch_size, embed_dim)

        # Classification
        x = self.head(x)  # (batch_size, num_classes)

        return x

def train_vit(config):
    # Extract config parameters
    patch_size = config['patch_size']
    embed_dim = config['embed_dim']
    depth = config['depth']
    num_heads = config['num_heads']
    mlp_ratio = config['mlp_ratio']

    print(f"Training ViT (patch_size={patch_size}, embed_dim={embed_dim}, depth/transformer layers={depth}, heads={num_heads}, MLP Ratio={mlp_ratio})")

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Load data
    train_loader, test_loader = get_cifar100_loaders(batch_size)

    # Initialize model
    model = VisionTransformer(
        img_size=32,
        patch_size=patch_size,
        in_channels=3,
        num_classes=100,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=0.1
    ).to(device)

    # Calculate parameters and FLOPs
    model_stats = summary(model, input_size=(batch_size, 3, 32, 32), verbose=0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"FLOPs per forward pass: {model_stats.total_mult_adds:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_start_time = time.time()
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })

        # Measure epoch time
        epoch_time = time.time() - epoch_start_time

        # Test accuracy
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Test Acc: {test_acc:.2f}%")

    # Calculate total training time
    total_training_time = time.time() - total_start_time

    # Save results
    config_str = f"ViT-p{patch_size}-e{embed_dim}-d{depth}-h{num_heads}"
    results = {
        'model': config_str,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'total_params': total_params,
        'flops': model_stats.total_mult_adds,
        'total_training_time': total_training_time,
        'avg_epoch_time': total_training_time / num_epochs,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': test_accuracies[-1]
    }

    return results

#%%%%%%%%%%%%%%%%%%%%%%%%%%Problem 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fine_tune_swin(model_name, num_epochs=5):
    print(f"Fine-tuning {model_name} on CIFAR-100")

    # Hyperparameters
    batch_size = 32
    learning_rate = 2e-5

    # Load data
    train_loader, test_loader = get_cifar100_swin_loaders(batch_size)

    # Initialize model
    model = SwinForImageClassification.from_pretrained(
        model_name,
        num_labels=100,
        ignore_mismatched_sizes=True
    ).to(device)

    # Freeze backbone for fine-tuning
    for param in model.swin.parameters():
        param.requires_grad = False

    # Calculate parameters and FLOPs
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get a sample input for FLOPs calculation
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    model_stats = summary(model, input_data=sample_input, verbose=0)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"FLOPs per forward pass: {model_stats.total_mult_adds:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Training loop
    total_start_time = time.time()
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })

        # Measure epoch time
        epoch_time = time.time() - epoch_start_time

        # Test accuracy
        test_acc = evaluate_swin_model(model, test_loader)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Test Acc: {test_acc:.2f}%")

    # Calculate total training time
    total_training_time = time.time() - total_start_time

    # Get model name (tiny or small)
    model_size = "tiny" if "tiny" in model_name else "small"

    # Save results
    results = {
        'model': f"Swin-{model_size}-pretrained",
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': model_stats.total_mult_adds,  # Add FLOPs to results
        'total_training_time': total_training_time,
        'avg_epoch_time': total_training_time / num_epochs,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': test_accuracies[-1]
    }

    return results

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)

        # Reshape to (B, H, W, C)
        x = x.view(B, H, W, C)

        # Group 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)

        # Concatenate along feature dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)

        # Flatten H and W
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)

        # Apply normalization and reduction
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2*C)

        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (window_height, window_width)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Define relative position bias table with a maximum size
        # We'll use a larger size to accommodate different window sizes
        max_window_size = 8  # Maximum window size we'll support
        self.max_window_size = max_window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_window_size - 1) * (2 * max_window_size - 1), num_heads)
        )

        # Initialize bias table
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Keep track of the current relative position index window size
        self.current_window_size = window_size

        # Initialize relative position indices for the initial window size
        self._init_rel_pos_index()

    def _init_rel_pos_index(self):
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        # Calculate relative coordinates between each pair of tokens
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        # Shift to start from 0
        relative_coords[:, :, 0] += self.max_window_size - 1  # Shift using max_window_size
        relative_coords[:, :, 1] += self.max_window_size - 1
        relative_coords[:, :, 0] *= 2 * self.max_window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)
        # Update the tracked window size
        self.current_window_size = self.window_size

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        # Check if window size has changed, if so recalculate relative position index
        if self.window_size != self.current_window_size:
            self._init_rel_pos_index()

        # QKV projection: (B*num_windows, window_size*window_size, 3*dim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*num_windows, num_heads, window_size*window_size, head_dim)

        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B*num_windows, num_heads, window_size*window_size, head_dim)

        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B*num_windows, num_heads, window_size*window_size, window_size*window_size)

        # Get appropriate relative position bias
        # We need to ensure the bias matches the current window size
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )  # window_size*window_size, window_size*window_size, num_heads

        # Permute to match attn dimensions
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, window_size*window_size, window_size*window_size

        # Add bias to attention scores
        attn = attn + relative_position_bias.unsqueeze(0)  # This should now have compatible dimensions

        # Apply mask if provided
        if mask is not None:
            # Convert mask to float
            nW = mask.shape[0]  # num_windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # Apply softmax
        attn = attn.softmax(dim=-1)

        # Apply attention to V
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B*num_windows, window_size*window_size, dim)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        # Ensure shift size is less than window size
        self.shift_size = min(shift_size, window_size // 2) if shift_size > 0 else 0
        self.mlp_ratio = mlp_ratio

        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Window Attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            dropout=dropout
        )

        # Attention mask for SW-MSA (for shifted windows)
        self.register_buffer("attn_mask", None)

    def forward(self, x):
        B, L, C = x.shape
        # Compute the side length of the feature map, assuming it's square
        H = W = int(L ** 0.5)

        # Ensure H and W are integers (L must be a perfect square)
        assert H * W == L, f"Input length {L} is not a perfect square, cannot reshape to square feature map"

        # Re-validate if current window size is appropriate for this feature map size
        if H < self.window_size:
            # Adjust window size dynamically if feature map is smaller than window_size
            # This is a safety check in case SwinTransformerStage didn't adjust it
            self.window_size = H
            self.shift_size = min(self.shift_size, self.window_size // 2)
            # Need to also adjust WindowAttention's window_size
            self.attn.window_size = self.window_size

        # Store shortcut connection
        shortcut = x

        # Apply first normalization
        x = self.norm1(x)

        # Reshape to (B, H, W, C) for spatial operations
        x = x.reshape(B, H, W, C)

        # Cyclic shift (for SW-MSA)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # Calculate attention mask (only needed for SW-MSA)
            if self.attn_mask is None or self.attn_mask.size(0) != ((H // self.window_size) * (W // self.window_size)):
                # Calculate mask for SW-MSA
                img_mask = torch.zeros((1, H, W, 1), device=x.device)
                h_slices = (slice(0, -self.window_size),
                           slice(-self.window_size, -self.shift_size),
                           slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                           slice(-self.window_size, -self.shift_size),
                           slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                # Windows partition on mask
                mask_windows = window_partition(img_mask, self.window_size)  # (num_windows*B, window_size, window_size, 1)
                mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)  # (num_windows*B, window_size*window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_windows*B, window_size*window_size, window_size*window_size)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                self.attn_mask = attn_mask
        else:
            shifted_x = x
            self.attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, window_size, window_size, C)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # (num_windows*B, window_size*window_size, C)

        # W-MSA/SW-MSA - note that self.attn now returns only x, not (x, q, k, v)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (num_windows*B, window_size*window_size, C)

        # Merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H_pad, W_pad, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Reshape back to (B, H*W, C)
        x = x.reshape(B, H * W, C)

        # First residual connection
        x = shortcut + x

        # Second normalization and MLP
        x = x + self.mlp(self.norm2(x))

        return x

class SwinTransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for i in range(depth)
        ])
        self.window_size = window_size

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)

        # if spatial dims < 2×2, just pass through
        if H <= 1:
            return x

        # figure out what window size we *should* be using
        effective_ws = min(self.window_size, H)

        # **Update every block’s window_size *and* its attention module** if it’s changed**
        if effective_ws != self.window_size:
            for block in self.blocks:
                block.window_size = effective_ws
                if block.shift_size > 0:
                    block.shift_size = effective_ws // 2

                # propagate into the attention module
                block.attn.window_size = effective_ws
                # force it to rebuild its relative_position_index
                block.attn.current_window_size = effective_ws
                block.attn._init_rel_pos_index()

        # now run through the blocks
        for block in self.blocks:
            x = block(x)

        return x


class BasicSwinTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=2,  # Smaller patch size for 32x32 images
        in_channels=3,
        num_classes=100,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,  # Smaller window size for 32x32 images
        mlp_ratio=4.,
        dropout=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.img_size = img_size

        # Check that window size can evenly divide the image patches
        patches_per_side = img_size // patch_size
        assert window_size <= patches_per_side, f"Window size ({window_size}) must be <= patches per side ({patches_per_side})"
        assert patches_per_side % window_size == 0, f"Patches per side ({patches_per_side}) must be divisible by window size ({window_size})"

        # Calculate the number of patches
        self.patches_resolution = patches_per_side
        self.num_patches = self.patches_resolution ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Check how many downsamplings we can do before resolution becomes too small
        # This ensures we don't create layers that would operate on feature maps smaller than window_size
        max_layers = 0
        curr_resolution = patches_per_side
        for i in range(len(depths)):
            if i > 0:  # After the first layer, we downsample
                curr_resolution = curr_resolution // 2
                if curr_resolution < 2:  # Stop if resolution gets too small
                    break
            max_layers += 1

        self.max_layers = max_layers

        # Reduce depths array if necessary to match max_layers
        if max_layers < len(depths):
            print(f"Warning: Reducing model depth from {len(depths)} to {max_layers} layers due to resolution constraints")
            depths = depths[:max_layers]
            num_heads = num_heads[:max_layers]

        # Layers
        self.stages = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        # Current feature resolution
        curr_resolution = self.patches_resolution

        # Feature dimension at each stage
        curr_dim = embed_dim

        # Build stages - only up to max_layers
        for i_layer in range(len(depths)):
            # Create stage
            stage = SwinTransformerStage(
                dim=curr_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            self.stages.append(stage)

            # Add patch merging layer except for the last stage
            if i_layer < len(depths) - 1:
                # We need enough resolution to keep merging
                if curr_resolution >= 2:
                    print(f"Warning: Resolution too small to merge at stage {i_layer}")
                    # Skip merging, just adjust dimension
                    #merge_layer = nn.Linear(curr_dim, curr_dim * 2)
                    merge_layer = PatchMerging(dim=curr_dim)
                    curr_resolution //= 2  # Halve the resolution
                    curr_dim *= 2  # Double the feature dimension
                else:
                    merge_layer = nn.Linear(curr_dim, curr_dim * 2)
                    curr_dim *= 2  # Double the feature dimension

                self.patch_merging_layers.append(merge_layer)

        # Final normalization and classification head
        self.norm = nn.LayerNorm(curr_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(curr_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: [B, C, H, W]

        # Patch embedding: [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.patch_embed(x)

        # Reshape for transformer: [B, embed_dim, H', W'] -> [B, H'*W', embed_dim]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Go through stages and patch merging layers
        for i, stage in enumerate(self.stages):
            # Apply transformer stage
            x = stage(x)  # [B, H*W, C]

            # Apply patch merging if not the last stage
            if i < len(self.patch_merging_layers):
                # Check if we need special handling for small feature maps
                curr_size = int((x.shape[1]) ** 0.5)

                if curr_size <= 2 and isinstance(self.patch_merging_layers[i], PatchMerging):
                    # Skip patch merging for very small feature maps, just use linear projection
                    lin = nn.Linear(x.shape[2], x.shape[2] * 2, device=x.device)
                    x = lin(x)
                else:
                    # Apply regular patch merging or linear layer
                    x = self.patch_merging_layers[i](x)

        # Normalization
        x = self.norm(x)  # [B, H*W, C]

        # Global pooling
        x = x.transpose(1, 2)  # [B, C, H*W]
        x = self.avgpool(x).flatten(1)  # [B, C]

        # Classification head
        x = self.head(x)  # [B, num_classes]

        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    # Ensure H and W are divisible by window_size
    assert H % window_size == 0, f"Height {H} not divisible by window size {window_size}"
    assert W % window_size == 0, f"Width {W} not divisible by window size {window_size}"

    # Reshape to group window pixels
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    # Permute and reshape to get windows
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    # Check that H and W are divisible by window_size
    assert H % window_size == 0, f"Height {H} not divisible by window size {window_size}"
    assert W % window_size == 0, f"Width {W} not divisible by window size {window_size}"

    # Calculate batch size
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # Reshape back to original image format
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x

def train_swin_from_scratch():
    print("Training Swin Transformer from scratch on CIFAR-100")

    # Hyperparameters
    batch_size = 32
    num_epochs = 5
    learning_rate = 2e-5

    # Load data
    train_loader, test_loader = get_cifar100_loaders(batch_size)

    # Initialize model with parameters optimized for CIFAR-100's small image size
    model = BasicSwinTransformer(
        img_size=32,
        patch_size=2, # Smaller patch size for 32x32 images
        in_channels=3,
        num_classes=100,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,  # Smaller window size for 32x32 images
        mlp_ratio=4.
    ).to(device)

    print(f"Swin Transformer architecture:")
    print(f"- Patch size: 2")
    print(f"- Window size: 4")
    print(f"- Patches resolution: {model.patches_resolution}")
    print(f"- Number of patches: {model.num_patches}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Get a sample input for FLOPs calculation
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    #sample_input = torch.randn(batch_size, 3, 32, 32).to(device)
    model_stats = summary(model, input_data=sample_input, verbose=0)
    print(f"Total parameters: {total_params:,}")
    print(f"FLOPs per forward pass: {model_stats.total_mult_adds:,}")

    # Loss and optimizer (train all parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_start_time = time.time()
    test_accuracies = []

    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })

        # Measure epoch time
        epoch_time = time.time() - epoch_start_time

        # Test accuracy
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s - Test Acc: {test_acc:.2f}%")

    # Calculate total training time
    total_training_time = time.time() - total_start_time

    # Save results
    results = {
        'model': 'Swin-from-scratch',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': model_stats.total_mult_adds,
        'total_training_time': total_training_time,
        'avg_epoch_time': total_training_time / num_epochs,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': test_accuracies[-1]
    }

    return results

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy

def evaluate_swin_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).logits
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy

def run_problem1_training_best():
    print("Running experiments for Problem 1: ResNet-18 vs ViT")

    results = []

    # ResNet-18 baseline
    resnet_results = train_resnet18()
    results.append(resnet_results)

    # ViT configurations
    vit_configs = [
        # Patch size, embed dim, depth, num_heads, mlp_ratio
        {'patch_size': 4, 'embed_dim': 256, 'depth': 4, 'num_heads': 4, 'mlp_ratio': 4},
        {'patch_size': 4, 'embed_dim': 512, 'depth': 4, 'num_heads': 8, 'mlp_ratio': 4},
        {'patch_size': 8, 'embed_dim': 256, 'depth': 4, 'num_heads': 4, 'mlp_ratio': 4},
        {'patch_size': 8, 'embed_dim': 512, 'depth': 8, 'num_heads': 8, 'mlp_ratio': 4},
    ]

    for config in vit_configs:
        vit_results = train_vit(config)
        results.append(vit_results)

    return results

def run_problem1_training_full():
    print("Running experiments for Problem 1: ResNet-18 vs ViT")

    results = []

    # ResNet-18 baseline
    resnet_results = train_resnet18()
    results.append(resnet_results)

     # ViT configurations to sweep
    patch_sizes  = [4, 8]
    embed_dims   = [256, 512]
    depths       = [4, 8]
    num_heads    = [2, 4]
    mlp_ratios   = [2, 4]

    vit_configs = [
        {
            'patch_size': p,
            'embed_dim': e,
            'depth': d,
            'num_heads': h,
            'mlp_ratio': r
        }
        for p in patch_sizes
        for e in embed_dims
        for d in depths
        for h in num_heads
        for r in mlp_ratios
    ]

    for config in vit_configs:
        vit_results = train_vit(config)
        results.append(vit_results)

    return results

def run_problem2_training_full():
    print("Running experiments for Problem 2: Swin Transformer")

    results = []

    # Fine-tune Swin-Tiny
    swin_tiny_results = fine_tune_swin("microsoft/swin-tiny-patch4-window7-224")
    results.append(swin_tiny_results)

    # Fine-tune Swin-Small
    swin_small_results = fine_tune_swin("microsoft/swin-small-patch4-window7-224")
    results.append(swin_small_results)

    # Train Swin from scratch
    swin_scratch_results = train_swin_from_scratch()
    results.append(swin_scratch_results)

    return results

def print_results_table(results, title, csv_filename):
    print(f"\n{title}")
    print("=" * 130)

    # Collect rows
    table_rows = []
    for result in results:
        model_name = result['model']
        total_params = f"{result['total_params']:,}"
        flops = f"{result['flops']:,}" if 'flops' in result else "N/A"
        training_time = f"{result['total_training_time']:.2f}s"
        epoch_time = f"{result['avg_epoch_time']:.2f}s"
        final_acc = f"{result['final_test_accuracy']:.2f}%"

        test_accuracies = result.get('test_accuracies', [])
        if len(test_accuracies) >= 10:
            acc_10 = f"{test_accuracies[9]:.2f}%"
        elif test_accuracies:
            acc_10 = f"{max(test_accuracies):.2f}%"
        else:
            acc_10 = "N/A"

        row = {
            "Model": model_name,
            "Params": total_params,
            "FLOPs": flops,
            "Training Time": training_time,
            "Epoch Time": epoch_time,
            "Final Acc": final_acc,
            "10-Epoch Acc": acc_10
        }
        table_rows.append(row)

    # Convert to DataFrame and print
    df = pd.DataFrame(table_rows)
    print(df.to_string(index=False))
    print("=" * 130)

    if csv_filename:
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to: {csv_filename}")

def plot_test_accuracies(results, title, filename):
    plt.figure(figsize=(10, 6))

    # Plot each model’s accuracy curve
    for result in results:
        epochs = range(1, len(result['test_accuracies']) + 1)
        plt.plot(epochs, result['test_accuracies'], marker='o', label=result['model'])

    # Labels & grid
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Layout + legend outside to the right
    plt.tight_layout()
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize='small',
        title='Model',
        frameon=True
    )

    # Save with extra padding to include the legend
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Problem 1: ResNet-18 vs ViT Full (every combination == 32)
    p1_results_full = run_problem1_training_full()
    print_results_table(p1_results_full, "Problem 1 Results: ResNet-18 vs ViT Full", "results/run_problem1_training_full_results.csv")
    plot_test_accuracies(p1_results_full, "ResNet-18 vs ViT Test Accuracy Full", "results/resnet_vs_vit_full.png")

    #Problem 2: Swin Transformer
    p2_results = run_problem2_training_full()
    print_results_table(p2_results, "Problem 2 Results: Swin Transformer", "results/swin_results.csv")
    plot_test_accuracies(p2_results, "Swin Transformer Test Accuracy", "results/swin_results.png")

    print("\nHomework 6 Execution Complete")

if __name__ == "__main__":
    main()
