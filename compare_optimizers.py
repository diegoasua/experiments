import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

# Check if MPS is available
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Grid search parameters
learning_rates = np.logspace(-6, 1, 10)  # Changed from (-4, -1) to (-6, 1)
weight_decays = np.logspace(-5, -2, 10)

# Optimizer configurations
optimizer_configs = {
    'SGD+Momentum': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd),
    'Adam+L2': lambda params, lr, wd: optim.Adam(params, lr=lr, weight_decay=wd),
    'AdamW': lambda params, lr, wd: optim.AdamW(params, lr=lr, weight_decay=wd),
    'SGD+Momentum+WeightDecay': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
}

def train_one_epoch(model, optimizer, scheduler, criterion, train_loader):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step(batch_idx / num_batches)
        total_loss += loss.item()
    
    return total_loss / num_batches

# Results storage
results = {name: np.zeros((len(learning_rates), len(weight_decays))) 
          for name in optimizer_configs.keys()}

# Main experiment loop
criterion = nn.CrossEntropyLoss()

for opt_name, opt_fn in optimizer_configs.items():
    print(f"Testing {opt_name}")
    
    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            model = SimpleCNN().to(device)
            optimizer = opt_fn(model.parameters(), lr, wd)
            scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))
            
            avg_loss = train_one_epoch(model, optimizer, scheduler, criterion, train_loader)
            results[opt_name][i, j] = avg_loss

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, (opt_name, result) in enumerate(results.items()):
    sns.heatmap(
        result,
        ax=axes[idx],
        xticklabels=[f'{wd:.1e}' for wd in weight_decays],
        yticklabels=[f'{lr:.1e}' for lr in learning_rates],
        cmap='viridis',
        cbar_kws={'label': 'Loss'}
    )
    axes[idx].set_title(opt_name)
    axes[idx].set_xlabel('Weight Decay')
    axes[idx].set_ylabel('Learning Rate')

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.close()