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

# Multi-Head Attention (same as before)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_v)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_probs, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)


# Memory Network - Attention over external memory
class MemoryNetwork(nn.Module):
    def __init__(self, embedding_dim, memory_size, hidden_dim): # Add hidden_dim as an argument
        super(MemoryNetwork, self).__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim # Store hidden_dim
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        # Linear layer to project encoder output to embedding dimension
        self.linear = nn.Linear(hidden_dim * 2, embedding_dim) # Project to embedding_dim


    def forward(self, query):
        # Project query to embedding dimension before computing similarity
        query = self.linear(query) # Project query to correct dimensions

        # Compute similarity between query and memory
        sim = torch.matmul(query, self.memory.T)  # Cosine similarity between query and memory
        attention_weights = F.softmax(sim, dim=-1)

        # Weighted sum of memory
        memory_output = torch.matmul(attention_weights, self.memory)

        return memory_output


# Gated Residual Networks (GRN)
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        gate = self.sigmoid(x)
        return residual + gate * (x - residual)  # Gated residual connection


# Ultimate Encoder-Decoder Model
class UltimateEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=512, num_classes=2, num_heads=8, num_layers=6, memory_size=128):
        super(UltimateEncoderDecoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim)
        self.attn = MultiHeadAttention(d_model=embedding_dim, num_heads=num_heads)
        self.memory_network = MemoryNetwork(embedding_dim, memory_size, hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim * 2, hidden_dim * 2)

        # Decoder (Transformer Layer)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Feedforward layer for output
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        # Apply positional encoding and attention
        x = self.positional_encoding(x)
        attn_output = self.attn(x, x, x)

        # Encoder: Bidirectional LSTM
        encoder_out, _ = self.encoder(attn_output)

        # Memory Network to fetch important information
        memory_output = self.memory_network(encoder_out)

        # Combine encoder output and memory output
        combined = encoder_out + memory_output
        grn_output = self.grn(combined)

        # Decoder (using Transformer decoder)
        decoder_out = self.decoder(grn_output, grn_output)

        # Use the last decoder output for classification
        return decoder_out[-1, :, :]

    def forward(self, q1, q2):
        # Process both question pairs through the encoder-decoder model
        out1 = self.forward_once(q1)
        out2 = self.forward_once(q2)

        # Concatenate the outputs of both questions and classify
        combined = torch.cat([out1, out2], dim=1)
        return self.fc(combined)
    
model = UltimateEncoderDecoder(embedding_dim=300, num_heads=6, num_layers=6, memory_size=128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, DEVICE)
evaluate_model(model, test_loader, DEVICE)