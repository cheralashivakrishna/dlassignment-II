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

# 🔤 English-to-Hindi Character-Level Transliteration  
**Using Sequence-to-Sequence LSTM (Encoder-Decoder)**  
Course: `22CAC04` — Deep Learning  
Institute: Chaitanya Bharathi Institute of Technology  

## 📌 Project Description
This project implements a **character-level sequence-to-sequence transliteration model** that converts Latin script (English characters) into Devanagari script (Hindi). It uses an encoder-decoder architecture with LSTM units and greedy decoding for inference. The dataset used is the Hindi portion of the **Dakshina dataset**.

## 📂 Dataset
**Dakshina Dataset (Hindi)** from AI4Bharat:  
- Train file: `hi.translit.sampled.train.tsv`  
- Dev file: `hi.translit.sampled.dev.tsv`  
- Test file: `hi.translit.sampled.test.tsv`  

Each line contains:  
```
<Devanagari Word> <Latin Word> <Count>
```

## 📚 Architecture Overview
- **Encoder**: LSTM with an embedding layer to process the Latin script input.
- **Decoder**: LSTM with an embedding and dense layer to generate Devanagari output.
- **Loss Function**: Sparse categorical crossentropy.
- **Decoder Strategy**: Greedy decoding (token with max probability at each time step).

## 🧠 Model Details
| Component   | Details |
|------------|---------|
| Embedding Dim | 256 |
| Hidden Units  | 512 |
| Encoder       | LSTM |
| Decoder       | LSTM |
| Output Layer  | Dense with softmax |
| Parameters    | ~3.2 Million |
| Accuracy      | ~98.84% (Training) |
| Val Accuracy  | ~94.88% |

## 🏋️ Training
- **Epochs**: 10  
- **Batch Size**: 64  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  

Example training log:
```
Epoch 1/10 - loss: 1.0974 - accuracy: 0.7296 - val_loss: 0.6642 - val_accuracy: 0.8114
...
Epoch 10/10 - loss: 0.0377 - accuracy: 0.9884 - val_loss: 0.1842 - val_accuracy: 0.9488
```

## 🔍 Inference Output (Greedy Decoding)
| Input (Latin) | Predicted (Devanagari) |
|---------------|-------------------------|
| ank           | अंक                     |
| anka          | अंका                    |
| ankit         | अंकित                   |
| anakon        | अनकों                   |
| ankhon        | अनखों                   |
| ankon         | अंकों                   |
| angkor        | अंकोकर                  |
| ankor         | अंकोर                   |
| angaarak      | अंगारक                  |
| angarak       | अंगारक                  |

## 🛠 How to Run

1. Place the `.tsv` files in your working directory or Colab `/content/`.
2. Run the full Python script.
3. Evaluate with `decode_sequence()` on test inputs.

## ✅ Future Improvements
- Add **Attention Mechanism** for better alignment.
- Implement **Beam Search** decoding.
- Train with **Subword or Word-level** units for larger vocabularies.
- Integrate **BLEU Score** or **Edit Distance** for evaluation.
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
