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

# Step 4: Define the Baseline Neural Network
class BaselineNN(nn.Module):
    def __init__(self, embedding_dim=300):
        super(BaselineNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, q1, q2):
        # Concatenate embeddings along the feature axis
        combined = torch.cat([q1.mean(dim=1), q2.mean(dim=1)], dim=1)
        return self.fc(combined)
    
baseline_nn_model = BaselineNN(embedding_dim=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_nn_model.parameters(), lr=0.001)

# Train and Evaluate
train_model(baseline_nn_model, train_loader, criterion, optimizer, DEVICE, epochs=2)
evaluate_model(baseline_nn_model, test_loader, DEVICE)