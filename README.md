# 2-way-1-shot-classification-nlp-model-

# 🧾 Fine Appeal Likelihood Classifier

This repository contains a prototype binary text classifier that predicts whether a traffic fine is likely to be **appealed** (`1`) or **not appealed** (`0`), based on the **full OCR-parsed text of Italian traffic fines**.

The model is inspired by **one-shot learning** principles and is designed for **low-data environments**, where only a handful of examples are available. It serves as a lightweight proof of concept for using language models to support document triage in legal or bureaucratic contexts.

---

## 📌 What This Project Uses

- **Language Embedding**: [`SentenceTransformer`](https://www.sbert.net) with the `all-MiniLM-L6-v2` model
- **Framework**: `PyTorch`
- **Embedding Strategy**: Document-level embeddings using mean pooling
- **Classifier**: A shallow neural network (1 hidden layer with ReLU + dropout)
- **Loss Function**: `BCELoss` (Binary Cross Entropy Loss)
- **Evaluation**: Logs training loss over 10 epochs and infers on a held-out example
- **Hardware Support**: CUDA/GPU support if available

---

## 🧾 Example Input Format

Each item is a tuple:
- Full **fine text** (from OCR or manual extraction)
- **Label**: `0` (not appealed) or `1` (appealed)

Model Architecture :
SentenceTransformer ('all-MiniLM-L6-v2') → 384-dim vector
       ↓
Feedforward Neural Net:
  Linear(384 → 128) → ReLU → Dropout(0.2) → Linear(128 → 1) → Sigmoid
       ↓
Predicted Probability (Appeal Likelihood)


