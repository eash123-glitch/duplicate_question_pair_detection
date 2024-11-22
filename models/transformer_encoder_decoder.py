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

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_classes=2, num_heads=6, num_layers=6):
        super(TransformerEncoderDecoder, self).__init__()
        # Transformer Encoder and Decoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        # Fully connected layer for classification
        # The input dimension of the first linear layer should match the concatenated output
        # which is 2 * embedding_dim (300 from each of out1 and out2)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # Changed input dimension to embedding_dim * 2
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        # Transformer requires input in the shape (sequence_length, batch_size, embedding_dim)
        x = x.transpose(0, 1)  # (batch_size, sequence_length, embedding_dim) -> (sequence_length, batch_size, embedding_dim)
        memory = self.encoder(x)  # Encoder processes the input sequence
        output = self.decoder(x, memory)  # Decoder processes the output of the encoder
        return output[-1, :, :]  # Use the last output for classification

    def forward(self, q1, q2):
        out1 = self.forward_once(q1)
        out2 = self.forward_once(q2)
        # Concatenate the outputs of both questions
        combined = torch.cat([out1, out2], dim=1)
        # Classify the similarity between the two questions
        return self.fc(combined)
    
model = TransformerEncoderDecoder(embedding_dim=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, DEVICE)
evaluate_model(model, test_loader, DEVICE )