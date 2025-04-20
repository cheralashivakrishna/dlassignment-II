# Deep Learning Assignment 2
**Course:** Deep Learning - 22CAC04  
**Institution:** Chaitanya Bharathi Institute of Technology  
**Department:** Information Technology  
**Due Date:** 20-04-25  

---

## 🔍 Overview
This repository contains the implementation of Question 1 and Question 2 of the Deep Learning Assignment 2.

---

## 📌 Question 1: Latin-to-Devanagari Transliteration

### 🚀 Objective
The objective of this project is to build a flexible RNN-based sequence-to-sequence (seq2seq) architecture for transliterating Latin script inputs to their corresponding Devanagari script representations. The model supports multiple RNN cell types: SimpleRNN, GRU, and LSTM, with tunable hyperparameters.

### 🗂️ Dataset
Dataset used: Dakshina Dataset (Google)  
Files used:
- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Each file contains the following columns:
- **Devanagari script**
- **Latin transliteration**
- **Frequency count**

### 🧱 Model Architecture
- **Embedding Layer** for both encoder and decoder
- **Encoder RNN** (LSTM / GRU / SimpleRNN) processes the Latin script input
- **Decoder RNN** (LSTM / GRU / SimpleRNN) generates the Devanagari script character-by-character using the final encoder state
- **Dense Layer** with softmax activation for character prediction

#### Flexibility:
- Embedding Dimension
- Hidden Units
- RNN Cell Type (`'lstm'`, `'gru'`, `'rnn'`)
- Number of Layers (extendable in the function)

### 🧮 Theoretical Analysis
#### a) Total Number of Computations
Let:
- `m` = embedding dimension
- `k` = hidden size
- `T` = sequence length
- `V` = vocabulary size

Total computations (approx):
- **Encoder**: O(T × (m×k + k²))
- **Decoder**: O(T × (m×k + k² + k×V))

#### b) Total Number of Parameters
- **Encoder LSTM**: 4 × (k×(k + m + 1))
- **Decoder LSTM**: 4 × (k×(k + m + 1))
- **Dense Output**: k × V
- **Embedding Layers**: V × m (for both encoder and decoder)

### 📊 Training Details
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 30
- **Validation Accuracy**: ~94.6%
- **Test Accuracy**: 0.9457

### 📈 Sample Predictions
| Input (Latin) | Target (Devanagari) | Predicted |
|---------------|---------------------|-----------|
| a n k         | अ ं क                | ऐंक      |
| a n k a       | अ ं क                | अंका      |
| a n k i t     | अ ं क ि त            | अंकित    |
| a n a k o n   | अ ं क ो ं            | अनकों    |
| a n k h o n   | अ ं क ो ं            | अंखों    |

### 🧲 Evaluation
- **Test Accuracy**: 0.9457

### 🛠️ How to Run
1. **Install Requirements**:
    ```bash
    pip install tensorflow==2.12.0 pandas gdown
    ```

2. **Run Training**:
    Ensure that the `.tsv` files from the Dakshina dataset are in your working directory.  
    Run the following script:
    ```bash
    python main_seq2seq_transliteration.py
    ```

---

## 🎶 Question 2: GPT-2 Fine-Tuning for Lyric Generation

### 🎯 Objective
To fine-tune a pre-trained GPT-2 language model on a dataset of English poetry/lyrics, enabling it to generate new song-like text sequences.

### 📁 Dataset
Used: Paul Timothy Mooney's Poetry Dataset (Kaggle)

- The `.txt` files were combined and cleaned.
- Each poem was appended with `<|endoftext|>` to mark the end of a sample.

### 🧪 Tokenization and Preprocessing
- Used `GPT2Tokenizer` with `eos_token` as the pad token.
- Applied max-length padding (512) and truncation.
- Prepared HuggingFace Dataset for training.
- Created `input_ids`, `attention_mask`, and `labels` for causal language modeling.

### 🧠 Model and Training Configuration
- **Model**: `GPT2LMHeadModel` (pretrained GPT-2)
- **Trainer** used from HuggingFace Transformers

#### Training Arguments:
- **Epochs**: 5
- **Batch Size**: 4
- **Learning Rate**: 5e-5
- **Output Directory**: `./lyrics_generator`
- Logging and saving every 500 steps

### 📉 Training Output
- **Training Loss**: 2.79
- **Train Runtime**: ~41 seconds for 5 epochs
- **Model saved to**: `./fine_tuned_lyrics_gpt2`

### 🎤 Sample Output
**Prompt**: _When the night comes_

**Generated Lyrics**:

When the night comes  
And the moon rises in it, to see us  
The dream that makes us fall  
And all that's to be forgotten  
The dream is coming true.

And you, like me, have told me all  
The tale of all this  
And you had this one word to tell of it all  
And this word is still the same  
And what do I know how to tell it all  
So tell me what is the true meaning of this dream

---

### 🛠️ How to Run
1. **Install Requirements**:
    ```bash
    pip install transformers datasets kagglehub
    ```

2. **Fine-Tune Model**:
    Run the following script:
    ```bash
    python gpt2_lyrics_finetune.py
    ```

---

## 📂 File Structure
.
├── README.md
├── main_seq2seq_transliteration.py  # All code for Q1
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv

---

## Question 2: GPT-2 Fine-Tuning for Lyric Generation
### Objective
The goal is to fine-tune a pre-trained GPT-2 language model on a dataset of English poetry/lyrics, enabling it to generate new song-like text sequences.

### Dataset
The dataset used is **Paul Timothy Mooney's Poetry Dataset** (available on Kaggle).

#### Files used:
- `poetry_dataset.txt`: Poetry dataset for Q2 (combined and cleaned)

### Tokenization and Preprocessing
- **Tokenizer**: GPT2Tokenizer with `<|endoftext|>` as the end-of-sequence token.
- **Padding**: Applied max-length padding (512) and truncation.
- **Dataset**: Prepared using HuggingFace’s `datasets` library.

### Model and Training Configuration
- **Model**: GPT2LMHeadModel (pretrained GPT-2)
- **Trainer**: HuggingFace Trainer
- **Training Arguments**:
  - **Epochs**: 5
  - **Batch Size**: 4
  - **Learning Rate**: 5e-5
  - **Output Directory**: `./lyrics_generator`

### Training Output
- **Training Loss**: 2.79
- **Training Time**: ~41 seconds for 5 epochs
- **Model Saved To**: `./fine_tuned_lyrics_gpt2`

### Sample Output
**Prompt:** When the night comes  
**Generated Lyrics:**
**When the night comes And the moon rises in it, to see us The dream that makes us fall And all that's to be forgotten The dream is coming true. And you, like me, have told me all The tale of all this And you had this one word to tell of it all And this word is still the same And what do I know how to tell it all So tell me what is the true meaning of this dream**

### How to Run
1. Install the required dependencies:
    ```bash
    pip install transformers datasets kagglehub
    ```
2. Fine-tune the GPT-2 model:
    ```bash
    python gpt2_lyrics_finetune.py
    ```

### File Structure for Question 2
📂 Question_2_Lyrics_Generation/
├── README.md                           # Documentation for Question 2
├── gpt2_lyrics_finetune.py             # Code for Q2 (GPT-2 fine-tuning for lyric generation)
├── data/                               # Folder containing dataset files for Q2
│   └── poetry_dataset.txt              # Dataset for Q2 (Poetry dataset)
├── models/                             # Folder to store trained models for Q2
│   └── fine_tuned_lyrics_gpt2/         # Fine-tuned GPT-2 model for Q2
│       ├── config.json                 # Configuration for the fine-tuned model
│       ├── pytorch_model.bin           # The model weights
│       └── vocab.json                  # Vocabulary file for the model
└── requirements.txt                    # File listing the dependencies for Q2



---

## Author
**Cheralashivakrishna**  


---

### References:
- Keras LSTM Seq2Seq Example
- Machine Learning Mastery - Seq2Seq
- GPT-2 Fine-tuning Tutorial
- Poetry Dataset on Kaggle

---

This `README.md` covers both questions with details of each task, code structure, how to run it,
