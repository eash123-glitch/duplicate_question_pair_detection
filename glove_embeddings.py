# Step 1: Load GloVe embeddings
import numpy as np

EMBEDDING_DIM = 300


def load_glove_embeddings(file_path, embedding_dim=300):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load GloVe
glove_path = "/content/glove.6B.300d.txt/glove.6B.300d.txt"  # Update this path
glove_embeddings = load_glove_embeddings(glove_path, EMBEDDING_DIM)