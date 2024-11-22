import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

BATCH_SIZE = 64
EMBEDDING_DIM = 300
MAX_LENGTH = 50
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train import train_model
from eval import evaluate_model
from tensor_creation import train_loader
from tensor_creation import test_loader

class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=300, filter_sizes=[3, 4, 5], num_filters=128, num_classes=2):
        super(SiameseCNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # Convolution + Maxpool layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes) * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply convolutions
        pooled_results = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in conv_results]  # Max-pooling
        return torch.cat(pooled_results, dim=1)  # Concatenate all features

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        # Concatenate the outputs from both branches and pass through the fully connected layers
        combined = torch.cat([out1, out2], dim=1)
        return self.fc(combined) # This line was added to return the output of the model

# Model, Loss, Optimizer
Siamese_CNN_model = SiameseCNN(embedding_dim=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Siamese_CNN_model.parameters(), lr=0.001)

# Train and Evaluate
train_model(Siamese_CNN_model, train_loader, criterion, optimizer, DEVICE, epochs=2)
evaluate_model(Siamese_CNN_model, test_loader, DEVICE)