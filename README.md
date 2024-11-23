# **Too many Questions**

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

### **Creating Tensors and Using GloVe Embeddings**

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
   - Glove embeddings can be downloaded by executing following command:
     ```bash
     pip install gdown
     gdown --id 16uAbJ0bgq3mXo64bpYWhMqmLJnNg33nUGLvEmzsSeGM -O output.txt
     ```

4. **Tensor Preparation**  
   - Question pairs are represented as two tensors (one for each question) of equal size.  
   - These tensors, along with their labels (`0` or `1`), are prepared as inputs for the model.

By leveraging **PyTorch** for tensor operations and GloVe for embeddings, this approach establishes a robust pipeline for converting text data into a numerical format that is well-suited for deep learning. The use of pre-trained GloVe vectors ensures that the model benefits from linguistic knowledge, improving its ability to understand the semantic relationships between question pairs.

### Implementation

To run the models, first go to models folder :
```bash
cd models
```

#### 1. **Baseline Neural Network**
The **BaselineNN** serves as a simple starting point. It directly takes the pre-trained GloVe embeddings of the two input questions, averages them to create a single feature vector for each question, and concatenates these vectors. The concatenated vector is then passed through a series of fully connected layers to classify whether the questions are duplicates or not.  

- Provides a baseline accuracy for comparison with more complex architectures.
- Lacks the ability to capture sequential dependencies or local patterns in the input data.
```bash
python3 baseline_nn.py
```
---

#### 2. **Siamese Convolutional Neural Network (SiameseCNN)**  
The **SiameseCNN** extracts local features from the input embeddings using convolutional layers with varying kernel sizes. For each question, the model applies convolutions followed by max pooling to capture n-gram patterns in the embeddings. The outputs of the two CNN branches are concatenated and passed through fully connected layers to predict the similarity.  

  - Effective in capturing local patterns like phrases or short n-grams.
  - Efficient due to parallel CNN layers.
  - Cannot model long-term dependencies across the sequence.
```bash
python3 siamese_cnn.py
```
---

#### 3. **Siamese Long Short-Term Memory Network (SiameseLSTM)**  
The **SiameseLSTM** uses LSTM layers to process the two input questions. LSTMs are effective in capturing sequential and contextual dependencies. The last hidden state of the bidirectional LSTM is used to represent each question. These representations are concatenated and passed through a fully connected layer for classification.

  - Captures long-term dependencies in questions.
  - Suitable for understanding semantic meaning in sequences.
  - Computationally more intensive compared to CNNs.
```bash
python3 siamese_lstm.py
```
---

#### 4. **Siamese LSTM with Convolutional Neural Network (SiameseLSTM-CNN)**  
The **SiameseLSTM-CNN** combines the strengths of LSTMs and CNNs. It uses LSTMs to capture sequential dependencies and CNNs to extract local features. For each question, the outputs from LSTM and CNN branches are concatenated to form a richer representation. These are then combined for classification.  

  - Leverages both sequential modeling (LSTM) and local feature extraction (CNN).
  - More expressive feature representations.
  - Increased complexity and training time due to dual processing branches.
```bash
python3 siamese_lstm_cnn.py
```
---

#### 5. **Siamese LSTM with Attention**  
The **SiameseLSTMWithAttention** introduces an attention mechanism to improve the semantic understanding of input sequences. Instead of relying solely on the last hidden state of the LSTM, it computes attention weights over all hidden states to focus on the most relevant parts of the sequence.  

  - Bidirectional LSTM captures forward and backward context.
  - Attention mechanism computes importance weights for each timestep.
  - Outputs an attention-weighted sum of LSTM hidden states, emphasizing key parts of the input.
  - Better at focusing on the most critical tokens in longer sequences.
  - Captures fine-grained semantic relationships between questions.
```bash
python3 lstm_with_attention.py
```
---

#### 6. **Simple Encoder-Decoder Model**  
This model uses an encoder-decoder architecture with LSTM layers. The encoder processes the input sequence and passes its output to the decoder, which generates a sequence representation. The last hidden state of the decoder is used for classification.

  - LSTM-based encoder-decoder design for sequence processing.
  - Sequential modeling of questions, with outputs used for similarity classification.
  - Simpler than attention-based models while effective for moderate-length sequences.
  - Provides a foundation for integrating attention in later models.
  - Relies only on the final hidden state of the decoder, potentially losing context for long sequences.
```bash
python3 simple_encoder_decoder.py
```
---

#### 7. **Bidirectional LSTM Encoder-Decoder (BiLSTMEncoderDecoder)**  
This model builds on the encoder-decoder architecture by introducing bidirectional LSTM layers in both the encoder and decoder. Bidirectional LSTMs allow the model to understand both past and future contexts, making the representations richer.

  - Bidirectional LSTM in both encoder and decoder.
  - Uses all context from both directions during encoding and decoding.
  - Outputs a concatenation of bidirectional decoder states for classification.
  - Stronger contextual modeling than unidirectional encoder-decoder models.
  - Captures dependencies from both ends of the sequence.
---

8. **Attention-based Encoder-Decoder (AttentionEncoderDecoder)**

The **Attention-based Encoder-Decoder** model enhances the basic encoder-decoder structure by adding an attention mechanism. This model allows the decoder to focus on different parts of the input sequence at each time step, improving its ability to generate more accurate outputs based on relevant context.

   - The attention mechanism enables the model to weigh different parts of the input sequence differently, allowing the decoder to focus on the most relevant tokens for generating the output.
   - The model can attend to the entire input sequence, not just relying on the final hidden state of the encoder. This allows it to capture dependencies between non-adjacent tokens more effectively.
   - *At each decoding step, the model selects relevant parts of the input sequence to inform its output, leading to more accurate and contextually-aware generation.
   - Unlike simple encoder-decoder models, attention-based models can handle longer sequences by allowing the decoder to access all parts of the input sequence, avoiding the limitations of using only the final hidden state.
   - The model’s complexity can lead to overfitting if not properly regularized, especially on smaller datasets.
---

9. **Complex Encoder-Decoder with Multi-Head Attention and Residual Connections (ComplexEncoderDecoder)**

The **Complex Encoder-Decoder** model takes the attention-based architecture a step further by integrating **multi-head attention** and **residual connections**, creating a more powerful structure for modeling complex dependencies in input sequences. 

   - By using multiple attention heads, the model is able to attend to different aspects of the input sequence in parallel. This allows for richer feature extraction and a better understanding of varied relationships within the sequence.
   - **Residual Connections**: These connections help mitigate the vanishing gradient problem by allowing gradients to flow more easily during backpropagation. They also enable the model to more effectively learn deeper representations.
   - **Positional Encoding**: Positional encoding is used to give the model a sense of token order, which is critical in sequence-based tasks like question answering and SPARQL query generation.
   - **Increased Complexity**: The combination of multi-head attention, residual connections, and positional encoding creates a highly expressive model that can handle complex tasks but at the cost of increased computational requirements and potential difficulty in tuning.
   - **Improved Performance**: By capturing long-range dependencies and incorporating multiple perspectives through attention heads, this model improves performance on tasks involving intricate patterns and contextual information.
```bash
python3 complex_transformer.py
```
---

10. **Transformer Encoder-Decoder (TransformerEncoderDecoder)**

The **Transformer Encoder-Decoder** model uses a purely attention-based architecture, completely removing recurrent layers (like LSTMs) in favor of self-attention mechanisms. This architecture is known for its scalability and efficiency in processing long sequences.

   - **Self-Attention**: The self-attention mechanism allows the model to compute attention scores for all tokens in the sequence, providing an efficient way to capture dependencies across long distances.
   - **No Recurrent Layers**: Unlike LSTM-based models, the transformer relies solely on attention layers, making it faster to train and more effective at parallelizing computations.
   - **Stacked Layers**: The model uses multiple layers of attention and feedforward networks in both the encoder and decoder, improving its ability to model complex relationships in the data.
   - **Scalability**: The transformer model is highly scalable, making it suitable for processing long sequences and large datasets, especially for tasks like translation or complex question answering.
   - **Computational Efficiency**: Although powerful, transformers can be computationally expensive in terms of memory and processing time, particularly with large-scale datasets or very long sequences.
```bash
python3 transformer_encoder_decoder.py
```
---
11. **UltimateEncoderDecoder with Gated Residual Networks and Memory (UltimateEncoderDecoder)**

The **UltimateEncoderDecoder** model combines the best features of all the previous models, incorporating **Gated Residual Networks (GRN)**, **Memory Networks**, and **Multi-Head Attention** to achieve the most powerful sequence-to-sequence architecture.

   - **Gated Residual Networks**: The introduction of GRNs enables better control over information flow, allowing the network to adjust the contribution of residual connections at each layer. This improves training stability and generalization.
   - **Memory Networks**: The addition of memory networks allows the model to retain and access important information from earlier steps in the sequence. This helps in tasks requiring long-term dependencies, such as generating SPARQL queries or detecting duplicate questions.
   - **Multi-Head Attention**: Multiple attention heads allow the model to focus on various aspects of the input sequence in parallel, further improving the richness of the learned features.
   - **Transformer-Style Decoder**: The decoder uses a transformer-like architecture, combining self-attention and feedforward layers to process the output sequence.
   - **Scalability and Flexibility**: The model is highly scalable and can handle long-range dependencies, making it effective for tasks involving large datasets or complex question-answering scenarios.
   - **Complexity and Training Time**: With the combination of all these advanced techniques, the model requires more computational resources, making it more expensive to train and fine-tune. It may also be prone to overfitting if the dataset is not sufficiently large.
```bash
python3 complex_transformer.py
```
---

| **Model**                       | **Unique Feature**                                 | **Accuracy (%)** | **Strengths**                                      |
|----------------------------------|----------------------------------------------------|------------------|----------------------------------------------------|
| **BaselineNN**                   | Simple feedforward neural network for classification | 72.21            | Easy to implement, good baseline for comparison    |
| **SiameseCNN**                  | Convolutional layers for feature extraction       | 81.53            | Good for extracting local features, fast to train |
| **SiameseLSTM**                 | Twin LSTM for sequence comparison                 | 74.65            | Effective for sequence comparison tasks            |
| **SiameseLSTM-CNN**             | Hybrid model combining LSTM and CNN for feature extraction | 78.88        | Captures both sequential and local features       |
| **SiameseLSTMWithAttention**     | Attention over LSTM hidden states                 | 80.70            | Emphasizes critical tokens, handles long sequences |
| **SimpleEncoderDecoder**        | LSTM-based sequence processing                    | 75.45            | Simple, effective for moderate-length sequences    |
| **BiLSTMEncoderDecoder**        | Bidirectional LSTMs in encoder and decoder        | 85-87            | Rich contextual understanding, bidirectional context |
| **AttentionEncoderDecoder**     | Attention mechanism for context-based generation  | 79.09            | Focuses on relevant parts of input, better for long sequences |
| **ComplexEncoderDecoder**       | Multi-head attention, residual connections        | 80.23            | Deeper representation, handles complex dependencies |
| **TransformerEncoderDecoder**   | Pure attention-based architecture, no LSTMs       | 81.62            | Scalable, efficient for long sequences and large datasets |
| **UltimateEncoderDecoder**      | Gated Residual Networks, Memory Networks, Multi-head Attention | 73.28        | Combines best features for rich, context-aware representations |



