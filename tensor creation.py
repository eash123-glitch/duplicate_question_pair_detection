import torch
from torch.utils.data import Dataset, DataLoader
from create_dataset import DuplicateQuestionsDataset
from glove_embeddings import glove_embeddings

BATCH_SIZE = 64
EMBEDDING_DIM = 300
MAX_LENGTH = 50
def load_data_from_tsv(file_path):
      data = []
      with open(file_path, 'r') as f:
          for line in f:
              elements = line.strip().split('\t')
              duplicate = int(elements[0])
              q1 = elements[1]
              q2 = elements[2]
              data.append((q1, q2, duplicate))
      return data


train_data = load_data_from_tsv('./data/train.full.tsv')

# Create the Dataset and DataLoader
dataset = DuplicateQuestionsDataset(train_data, glove_embeddings, max_length=MAX_LENGTH, embedding_dim=EMBEDDING_DIM)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the test data from the file
test_data = load_data_from_tsv('./data/test.full.tsv')

# Create the Dataset and DataLoader for the test set
test_dataset = DuplicateQuestionsDataset(test_data, glove_embeddings, max_length=MAX_LENGTH, embedding_dim=EMBEDDING_DIM)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)