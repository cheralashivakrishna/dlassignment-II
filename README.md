## 🧐 Deep Learning Assignment 2

**Course:** Deep Learning - 22CAC04\
**Institution:** Chaitanya Bharathi Institute of Technology\
**Department:** Information Technology\
**Due Date:** 20-04-25

### 🔍 Overview

This repository contains the implementation of **Question 1** and **Question 2** of the Deep Learning Assignment 2.

---

## 📌 Question 1: Latin-to-Devanagari Transliteration

### 🚀 Objective

To build a flexible RNN-based seq2seq architecture for transliterating Latin script inputs to their corresponding Devanagari script representations. The model supports multiple cell types: **SimpleRNN**, **GRU**, and **LSTM**, with tunable hyperparameters.

---

### 🗂️ Dataset

Dataset used: [Dakshina Dataset (Google)](https://github.com/google-research-datasets/dakshina)\
Files used:

- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Each file contains columns:

- Devanagari script
- Latin transliteration
- Frequency count

---

### 🧱 Model Architecture

1. **Embedding Layer** for both encoder and decoder
2. **Encoder RNN (LSTM / GRU / SimpleRNN)** - processes the Latin script input
3. **Decoder RNN (LSTM / GRU / SimpleRNN)** - generates the Devanagari script character-by-character using the final encoder state
4. **Dense Layer** with softmax activation for character prediction

**Flexibility:**

- Embedding Dimension
- Hidden Units
- RNN Cell Type (`'lstm'`, `'gru'`, `'rnn'`)
- Number of Layers (extendable in the function)

---

### 🧮 Theoretical Analysis

#### a) Total Number of Computations

Let:

- `m` = embedding dimension
- `k` = hidden size
- `T` = sequence length
- `V` = vocabulary size

Total computations (approx):\
Encoder: O(T × (m×k + k²))\
Decoder: O(T × (m×k + k² + k×V))

#### b) Total Number of Parameters

Encoder LSTM: 4 × (k×(k + m + 1))\
Decoder LSTM: 4 × (k×(k + m + 1))\
Dense Output: k × V\
Embedding Layers: V × m (each for encoder and decoder)

---

### 📊 Training Details

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Batch Size:** 64
- **Epochs:** 30
- **Validation Accuracy:** \~94.6%
- **Test Accuracy:** **0.9457**

---

### 📈 Sample Predictions

| Input (Latin) | Target (Devanagari) | Predicted |
| ------------- | ------------------- | --------- |
| a n k         | अ ं क               | ऐंक       |
| a n k a       | अ ं क               | अंका      |
| a n k i t     | अ ं क ि त           | अंकित     |
| a n a k o n   | अ ं क ो ं           | अनकों     |
| a n k h o n   | अ ं क ो ं           | अंखों     |

---

### 🧲 Evaluation

```bash
Test Accuracy: 0.9457
```

---

### 🛠️ How to Run

#### 🔧 Install Requirements

```bash
pip install tensorflow==2.12.0 pandas gdown
```

#### ▶️ Run Training

Ensure the `.tsv` files from Dakshina dataset are in your working directory.

```python
python main_seq2seq_transliteration.py
```

---

### 📂 File Structure

```
.
├── README.md
├── main_seq2seq_transliteration.py  # All code for Q1
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```

---

📘 References

- [Keras LSTM Seq2Seq Example](https://keras.io/examples/nlp/lstm_seq2seq/)
- [Machine Learning Mastery - Seq2Seq](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)

---

## 📌 Question 2: GPT-2 Lyrics Generation

## Project Description

This project fine-tunes the GPT-2 language model using Hugging Face's `transformers` library to generate English song lyrics. The training data is sourced from publicly available lyrics datasets and processed using Google Colab.

## Features

- Fine-tuned GPT-2 model for lyrics generation
- Simple interface to generate lyrics using custom prompts
- Training pipeline implemented using Hugging Face Trainer API

## Folder Structure

```
📁 gpt2-lyrics-generation/
├── 📁 data/               # Contains lyrics text files
├── 📁 model/              # Stores trained model and checkpoints
├── 📁 outputs/            # Generated lyrics outputs
```

## Getting Started

### Prerequisites

Ensure the following libraries are installed:

```bash
pip install transformers datasets
```

### Dataset Preparation

Place your lyrics `.txt` files in a folder (e.g., `data/`) and combine them into a single `lyrics.txt` file.

### Model Training

Refer to the source code or Colab notebook to run the training pipeline:

- Load and preprocess data
- Tokenize using GPT-2 tokenizer
- Fine-tune using Trainer API

### Text Generation

After training, use the `pipeline` API for text generation:

```python
from transformers import pipeline

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print(generator("Love is", max_length=100)[0]['generated_text'])
```

## Sample Output

```
Love is you I get to know the way that you think
But it's something that keeps me from wanting
That love I might get to know you just a little more (Girl, I'm waiting) Yeah, you're a really good girl (Girl, you're a really good girl) Yeah, you're a really good girl (Girl, you're a really good girl)
You're a really good girl (Girl, you're a really good girl)
You're a really
```

## Contribution

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Hugging Face for the Transformers library
- Google Colab for GPU support
- Dataset sources from data.world and Kaggle

---

**Author:** Cherala Shiva Krishna
