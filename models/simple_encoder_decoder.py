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

class SimpleEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_classes=2):
        super(SimpleEncoderDecoder, self).__init__()
        # Encoder: LSTM layer
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Decoder: LSTM layer
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        # Encode the input sequence
        encoder_out, _ = self.encoder(x)
        # Decode the last output of the encoder (we'll use the last hidden state as the context)
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
    
simple_encoder_decoder_model = SimpleEncoderDecoder(embedding_dim=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(simple_encoder_decoder_model.parameters(), lr=0.001)

# Train and Evaluate
train_model(simple_encoder_decoder_model, train_loader, criterion, optimizer, DEVICE, epochs=1)
evaluate_model(simple_encoder_decoder_model, test_loader, DEVICE)
