from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

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
    
# Optimizer configurations
optimizer_configs = {
    'SGD+Momentum': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd),
    'Adam+L2': lambda params, lr, wd: optim.Adam(params, lr=lr, weight_decay=wd),
    'AdamW': lambda params, lr, wd: optim.AdamW(params, lr=lr, weight_decay=wd),
    'SGD+Momentum+WeightDecay': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
}

def train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device):
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
        scheduler.step()  # Removed the parameter here
        total_loss += loss.item()
    
    return total_loss / num_batches

def run_configuration(args):
    opt_name, lr, wd, device, train_loader, criterion = args
    model = SimpleCNN().to(device)
    optimizer = optimizer_configs[opt_name](model.parameters(), lr, wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))
    avg_loss = train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device)  # Added device here
    return opt_name, lr, wd, avg_loss

if __name__ == '__main__':

    mp.set_start_method('spawn')

    # Check if GPU is available
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

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

    # Results storage
    results = {name: np.zeros((len(learning_rates), len(weight_decays))) 
            for name in optimizer_configs.keys()}

    # Main experiment loop
    criterion = nn.CrossEntropyLoss()

    # Create all configurations
    all_configs = [
        (opt_name, lr, wd, device, train_loader, criterion)  # <-- Now has all 6 required values
        for opt_name in optimizer_configs.keys()
        for lr in learning_rates
        for wd in weight_decays
    ]

    # Batch size for parallel processing (adjust based on GPU memory)
    batch_size = 64  # Run 8 models in parallel
    num_batches = (len(all_configs) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_configs))
        batch_configs = all_configs[start_idx:end_idx]
        
        # Run configurations in parallel
        with mp.Pool(processes=len(batch_configs)) as pool:
            batch_results = pool.map(run_configuration, batch_configs)
        
        # Store results
        for opt_name, lr, wd, loss in batch_results:
            i = np.where(learning_rates == lr)[0][0]
            j = np.where(weight_decays == wd)[0][0]
            results[opt_name][i, j] = loss

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