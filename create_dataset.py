import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import nltk
from collections import defaultdict
from tqdm import tqdm
from preprocess import tokenize
from glove_embeddings import glove_embeddings

# Hyperparameters
BATCH_SIZE = 64
EMBEDDING_DIM = 300
MAX_LENGTH = 50
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuplicateQuestionsDataset(Dataset):
    def __init__(self, data, glove_embeddings, max_length=50, embedding_dim=300):
        self.data = data
        self.glove_embeddings = glove_embeddings
        self.max_length = max_length
        self.embedding_dim = embedding_dim

    def pad_and_embed(self, tokens):
        embeddings = [self.glove_embeddings.get(token, np.zeros(self.embedding_dim)) for token in tokens]
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]
        else:
            pad_length = self.max_length - len(embeddings)
            embeddings.extend([np.zeros(self.embedding_dim)] * pad_length)
        return np.array(embeddings)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q1, q2, label = self.data[idx]
        q1_tokens = tokenize(q1)
        q2_tokens = tokenize(q2)
        q1_embed = self.pad_and_embed(q1_tokens)
        q2_embed = self.pad_and_embed(q2_tokens)
        return torch.tensor(q1_embed, dtype=torch.float32), torch.tensor(q2_embed, dtype=torch.float32), torch.tensor(int(label), dtype=torch.long)

