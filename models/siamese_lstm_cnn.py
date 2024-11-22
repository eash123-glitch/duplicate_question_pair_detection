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

class SiameseLSTMCNN(nn.Module):
    def __init__(self, embedding_dim=300, num_classes=2, hidden_dim=128, num_layers=2, filter_sizes=[3, 4, 5], num_filters=128):
        # Fix: Use super() correctly by passing the class and self
        super().__init__()
        # Remove the explicit type and object argument and the code will inherit correctly
        # Embedding layer (assuming you're passing tokenized indices)
        # self.embedding = nn.EmbeddingBag(embedding_dim, 300)
        # Instead of EmbeddingBag, use a Linear layer to project the embeddings
        self.embedding = nn.Linear(embedding_dim, 300)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=300, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # CNN Layers (for extracting local features)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, 300)) for fs in filter_sizes
        ])

        # Fully connected layer
        # Calculate the correct input size for the first linear layer
        lstm_output_size = hidden_dim * 2  # Bidirectional LSTM output size
        cnn_output_size = num_filters * len(filter_sizes)
        fc_input_size = lstm_output_size + cnn_output_size  # Combining LSTM and CNN features

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size * 2, 128),  # Multiply fc_input_size by 2 to account for concatenation of q1 and q2
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        # Embedding layer
        # embedded = self.embedding(x)
        # Apply the linear projection
        embedded = self.embedding(x)

        # LSTM processing
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Get the last LSTM output (considering bidirectional output)

        # CNN processing
        # Ensure x has the correct shape for CNN
        x = embedded.unsqueeze(1)  # Add channel dimension for CNN processing
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply convolutions
        pooled_results = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in conv_results]  # Max-pooling

        # Concatenate all CNN features
        cnn_out = torch.cat(pooled_results, dim=1)

        # Combine both LSTM and CNN outputs
        combined = torch.cat([lstm_out, cnn_out], dim=1)

        return combined

    def forward(self, q1, q2):
        out1 = self.forward_once(q1)
        out2 = self.forward_once(q2)
        # Concatenate the outputs from both branches and pass through the fully connected layers
        combined = torch.cat([out1, out2], dim=1)
        return self.fc(combined)  # Return the output of the model
    
# Model, Loss, Optimizer
Siamese_LSTM_CNN_model = SiameseLSTMCNN(embedding_dim=EMBEDDING_DIM, num_classes=2)  # Assuming binary classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Siamese_LSTM_CNN_model.parameters(), lr=0.001)

# Train and Evaluate
train_model(Siamese_LSTM_CNN_model, train_loader, criterion, optimizer, DEVICE, epochs=1)
evaluate_model(Siamese_LSTM_CNN_model, test_loader, DEVICE)
