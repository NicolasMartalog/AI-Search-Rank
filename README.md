# AI Search Rank

A BERT-based semantic search and question similarity model that finds similar questions using deep learning.

## Features
- Semantic search using BERT embeddings
- Question similarity detection
- Batch processing for faster inference
- GPU acceleration support
- Mixed precision training
- Early stopping and learning rate scheduling

## Installation 
```
git clone https://github.com/yourusername/ai-search-rank.git
cd ai-search-rank
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### 1. Data Preprocessing
Prepare your questions dataset in CSV format with columns: "id", "qid1", "qid2", "question1", "question2", "is_duplicate". 
I used the Quara Question Pairs dataset: https://www.kaggle.com/datasets/quora/question-pairs-dataset

Run preprocessing:
```
python src/preprocess.py
```

### 2. Training
Train the BERT-based similarity model:
```
python src/train.py
```

### 3. Inference
Find similar questions using the trained model:
```
python src/infer.py
```

## Model Architecture
- Base: BERT (bert-base-uncased)
- Additional layers:
  - Dropout (0.1)
  - Linear layers: 1536 → 256 → 64 → 1
  - ReLU activations
- Loss: Mean Squared Error (MSE)
- Optimizer: AdamW with linear warmup

## Performance Optimizations
- Batch processing for inference
- GPU acceleration
- Mixed precision training
- Parallel data loading
- Pre-tokenization caching
- Early stopping
- Learning rate scheduling

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- NumPy
- Pandas
- tqdm

