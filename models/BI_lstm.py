
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

class BiLSTMEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_classes=2):
        super(BiLSTMEncoderDecoder, self).__init__()
        # Encoder: Bidirectional LSTM layer
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Decoder: Bidirectional LSTM layer
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),  # Concatenated from bidirectional decoder outputs
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        # Encode the input sequence
        encoder_out, _ = self.encoder(x)
        # Decode the encoded sequence (using bidirectional output)
        decoder_out, _ = self.decoder(encoder_out)
        # Use the last hidden state of the decoder for classification
        return decoder_out[:, -1, :]

    def forward(self, q1, q2):
        # Process each question through the encoder-decoder
        out1 = self.forward_once(q1)
        out2 = self.forward_once(q2)
        # Concatenate the outputs of both questions
        combined = torch.cat([out1, out2], dim=1)
        # Classify the similarity between the two questions
        return self.fc(combined)
    
model = BiLSTMEncoderDecoder(embedding_dim=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, DEVICE)
evaluate_model(model, test_loader, DEVICE)


