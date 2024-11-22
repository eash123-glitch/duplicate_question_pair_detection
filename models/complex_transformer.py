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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert embed_size % self.num_heads == 0, "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = values.permute(2, 0, 1, 3)
        keys = keys.permute(2, 0, 1, 3)
        query = query.permute(2, 0, 1, 3)

        energy = torch.einsum("qnhd,knhd->qknh", [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=2)

        out = torch.einsum("qknh,vnhd->qknd", [attention, values]).permute(1, 2, 0, 3).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class ComplexEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_heads, num_layers):
        super(ComplexEncoderDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)

        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.attention = MultiHeadAttention(embed_size=hidden_dim, num_heads=num_heads)

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(x)

        # Decoder (uses the encoder output and LSTM hidden states)
        decoder_output, _ = self.decoder_lstm(x, (hidden, cell))

        # Attention mechanism
        attention_output = self.attention(decoder_output, encoder_output, decoder_output, mask=None)

        # Final output layer
        out = self.fc_out(attention_output)
        return out

# Example usage with dummy data
model = ComplexEncoderDecoder(embedding_dim=300, hidden_dim=256, num_classes=2, num_heads=8, num_layers=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, DEVICE, epochs=1)
evaluate_model(model, test_loader, DEVICE)