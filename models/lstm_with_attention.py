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

class SiameseLSTMWithAttention(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_classes=2):
        super(SiameseLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)  # Attention layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def attention_layer(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return torch.sum(attn_weights * lstm_out, dim=1)

    def forward_once(self, x):
        lstm_out, _ = self.lstm(x)
        attention_out = self.attention_layer(lstm_out)
        return attention_out

    def forward(self, q1, q2):
        out1 = self.forward_once(q1)
        out2 = self.forward_once(q2)
        combined = torch.cat([out1, out2], dim=1)
        return self.fc(combined)

lstm_attention_model = SiameseLSTMWithAttention(embedding_dim=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_attention_model.parameters(), lr=0.001)

# Train and Evaluate
train_model(lstm_attention_model, train_loader, criterion, optimizer, DEVICE)
evaluate_model(lstm_attention_model, test_loader, DEVICE)