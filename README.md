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



