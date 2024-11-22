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

class SiameseLSTM(nn.Module):
    def __init__(self, embedding_dim=300, num_classes=2, hidden_dim=128, num_layers=2):
        super(SiameseLSTM, self).__init__()

        # Initialize the embedding layer
        # Assuming you are supplying the input embeddings rather than indexing into them:
        self.embedding = nn.Linear(embedding_dim, 300) # changed to nn.Linear to match the embedding_dim and output size
        self.lstm = nn.LSTM(input_size=300, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        # Apply the embedding layer
        embedded = self.embedding(x) # Remove reshaping, apply embedding directly
        # embedded = embedded.reshape(x.size(0) // x.size(1) ,x.size(1) // 1, 300)  # changed to x.size(0) // x.size(1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        # Concatenate the final output from both directions (forward, backward)
        return lstm_out[:, -1, :]

    def forward(self, q1, q2):
        out1 = self.forward_once(q1)
        out2 = self.forward_once(q2)
        # Concatenate the outputs from both branches and pass through the fully connected layers
        combined = torch.cat([out1, out2], dim=1)
        return self.fc(combined)  # Return the output of the model
    
# Model, Loss, Optimizer
Siamese_LSTM_model = SiameseLSTM(embedding_dim=EMBEDDING_DIM, num_classes=2)  # Assuming binary classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Siamese_LSTM_model.parameters(), lr=0.001)

# Train and Evaluate
train_model(Siamese_LSTM_model, train_loader, criterion, optimizer, DEVICE)
evaluate_model(Siamese_LSTM_model, test_loader, DEVICE)
