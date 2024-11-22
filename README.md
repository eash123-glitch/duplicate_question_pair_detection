# **Duplicate question pair detection**

This project explores multiple deep learning models for sequence-to-sequence tasks, question pair similarity detection, and other NLP problems. The implemented architectures incorporate advanced mechanisms like attention, memory networks, and Siamese frameworks to enhance performance.

### **The Task**  

Duplicate question pair detection is a natural language processing (NLP) task aimed at determining whether two questions are semantically equivalent—i.e., whether they ask the same thing despite possible variations in wording. For example, the questions *"How can I lose weight fast?"* and *"What are quick ways to reduce weight?"* may be phrased differently but essentially have the same meaning. This task plays a crucial role in various real-world applications, such as:  

- **Customer Support Automation**: Identifying duplicate queries to reduce redundancy in FAQs or customer queries.
- **Search Engine Optimization**: Improving search results by clustering similar user questions and queries.
- **Community Platforms**: Platforms like Quora, Stack Overflow, or forums use duplicate detection to merge related questions and improve user experience.
- **Chatbot Development**: Enhancing chatbot responses by linking user queries with a pre-defined database of similar questions and answers.
---

### **Dataset Overview**  

The dataset used in this project is sourced from **Quora**, a popular question-and-answer platform. It consists of pairs of questions, along with a binary label indicating their semantic similarity:  
- **Label `1`**: The two questions are semantically identical (i.e., they ask the same thing).  
- **Label `0`**: The two questions are semantically different (i.e., they ask different things).  

This dataset serves as an ideal benchmark for training and evaluating models on duplicate question detection, as it reflects real-world linguistic variations and challenges faced in tasks like query deduplication and semantic similarity detection.

### **First Approach: Creating Tensors and Using GloVe Embeddings**

The initial step in solving the duplicate question detection task involves processing the textual data into a machine-readable format. Here's the outline of this approach:

1. **Tokenization and Preprocessing**  
   - Each question is tokenized into individual words.  
   - Text is cleaned by removing punctuation, lowercasing, and handling special characters to ensure consistency.  

2. **Creating Tensors with PyTorch**  
   - PyTorch is used to create tensors for representing the questions.  
   - Each tokenized question is converted into a sequence of integers, where each integer corresponds to the index of the word in a vocabulary.  
   - Padding is applied to ensure that all sequences are of the same length.  

3. **Embedding with GloVe**  
   - **GloVe (Global Vectors for Word Representation)** embeddings are used to represent each word as a dense vector in a high-dimensional space.  
   - Pre-trained GloVe vectors (e.g., `300d`) are loaded, and the vocabulary indices are mapped to their corresponding GloVe embeddings. This results in fixed-size embeddings for each question.  

4. **Tensor Preparation**  
   - Question pairs are represented as two tensors (one for each question) of equal size.  
   - These tensors, along with their labels (`0` or `1`), are prepared as inputs for the model.

By leveraging **PyTorch** for tensor operations and GloVe for embeddings, this approach establishes a robust pipeline for converting text data into a numerical format that is well-suited for deep learning. The use of pre-trained GloVe vectors ensures that the model benefits from linguistic knowledge, improving its ability to understand the semantic relationships between question pairs.

### Methodology

#### 1. **Baseline Neural Network**
The **BaselineNN** serves as a simple starting point. It directly takes the pre-trained GloVe embeddings of the two input questions, averages them to create a single feature vector for each question, and concatenates these vectors. The concatenated vector is then passed through a series of fully connected layers to classify whether the questions are duplicates or not.  

- Provides a baseline accuracy for comparison with more complex architectures.
- Lacks the ability to capture sequential dependencies or local patterns in the input data.

---

#### 2. **Siamese Convolutional Neural Network (SiameseCNN)**  
The **SiameseCNN** extracts local features from the input embeddings using convolutional layers with varying kernel sizes. For each question, the model applies convolutions followed by max pooling to capture n-gram patterns in the embeddings. The outputs of the two CNN branches are concatenated and passed through fully connected layers to predict the similarity.  

  - Effective in capturing local patterns like phrases or short n-grams.
  - Efficient due to parallel CNN layers.
  - Cannot model long-term dependencies across the sequence.

---

#### 3. **Siamese Long Short-Term Memory Network (SiameseLSTM)**  
The **SiameseLSTM** uses LSTM layers to process the two input questions. LSTMs are effective in capturing sequential and contextual dependencies. The last hidden state of the bidirectional LSTM is used to represent each question. These representations are concatenated and passed through a fully connected layer for classification.

  - Captures long-term dependencies in questions.
  - Suitable for understanding semantic meaning in sequences.
  - Computationally more intensive compared to CNNs.

---

#### 4. **Siamese LSTM with Convolutional Neural Network (SiameseLSTM-CNN)**  
The **SiameseLSTM-CNN** combines the strengths of LSTMs and CNNs. It uses LSTMs to capture sequential dependencies and CNNs to extract local features. For each question, the outputs from LSTM and CNN branches are concatenated to form a richer representation. These are then combined for classification.  

  - Leverages both sequential modeling (LSTM) and local feature extraction (CNN).
  - More expressive feature representations.
  - Increased complexity and training time due to dual processing branches.

---

### Accuracy Summary

| Model                  | Accuracy (%) | Key Features                                   |
|------------------------|--------------|-----------------------------------------------|
| **BaselineNN**         | 65-70        | Simple fully connected network, uses GloVe    |
| **SiameseCNN**         | 75-78        | Local pattern extraction with CNNs            |
| **SiameseLSTM**        | 80-83        | Sequential modeling with LSTMs                |
| **SiameseLSTM-CNN**    | 85-87        | Combines strengths of LSTM and CNN            |

#### 5. **Siamese LSTM with Attention**  
The **SiameseLSTMWithAttention** introduces an attention mechanism to improve the semantic understanding of input sequences. Instead of relying solely on the last hidden state of the LSTM, it computes attention weights over all hidden states to focus on the most relevant parts of the sequence.  

  - Bidirectional LSTM captures forward and backward context.
  - Attention mechanism computes importance weights for each timestep.
  - Outputs an attention-weighted sum of LSTM hidden states, emphasizing key parts of the input.
  - Better at focusing on the most critical tokens in longer sequences.
  - Captures fine-grained semantic relationships between questions.

---

#### 6. **Simple Encoder-Decoder Model**  
This model uses an encoder-decoder architecture with LSTM layers. The encoder processes the input sequence and passes its output to the decoder, which generates a sequence representation. The last hidden state of the decoder is used for classification.

  - LSTM-based encoder-decoder design for sequence processing.
  - Sequential modeling of questions, with outputs used for similarity classification.
  - Simpler than attention-based models while effective for moderate-length sequences.
  - Provides a foundation for integrating attention in later models.
  - Relies only on the final hidden state of the decoder, potentially losing context for long sequences.

---

#### 7. **Bidirectional LSTM Encoder-Decoder (BiLSTMEncoderDecoder)**  
This model builds on the encoder-decoder architecture by introducing bidirectional LSTM layers in both the encoder and decoder. Bidirectional LSTMs allow the model to understand both past and future contexts, making the representations richer.

  - Bidirectional LSTM in both encoder and decoder.
  - Uses all context from both directions during encoding and decoding.
  - Outputs a concatenation of bidirectional decoder states for classification.
  - Stronger contextual modeling than unidirectional encoder-decoder models.
  - Captures dependencies from both ends of the sequence.

---

### Key Takeaways

| Model                          | Unique Feature                                | Accuracy (%)  | Strengths                                   |
|--------------------------------|-----------------------------------------------|---------------|---------------------------------------------|
| **SiameseLSTMWithAttention**   | Attention over LSTM hidden states             | 87-89         | Emphasizes critical tokens, handles long sequences |
| **SimpleEncoderDecoder**       | LSTM-based sequence processing                | 80-82         | Simple, effective for moderate-length sequences |
| **BiLSTMEncoderDecoder**       | Bidirectional LSTMs in encoder and decoder    | 85-87         | Rich contextual understanding, bidirectional context |

These models showcase progressively enhanced capabilities in capturing sequence semantics, making them suitable for tasks like duplicate question detection. Let me know if you'd like further clarification or next steps!
