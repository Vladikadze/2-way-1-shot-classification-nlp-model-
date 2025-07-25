# 2-way-1-shot-classification-nlp-model-

# 🧾 Fine Appeal Likelihood Classifier

I have a side project on few shot learning so I wanted to experiment with it. 

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

## 🔧 Model Architecture

SentenceTransformer ('all-MiniLM-L6-v2') → 384-dim vector
↓
Feedforward Neural Net:
Linear(384 → 128) → ReLU → Dropout(0.2) → Linear(128 → 1) → Sigmoid
↓
Predicted Probability (Appeal Likelihood)


---

I tested it on paper number 2,  
and these were the results:  
**Appeal likelihood: 0.4957**  
**The fine is unlikely to be appealed.**

---

As input data, I used the papers that were available to me.  
- Label `1`: Paper 7 (which had an appeal linked to it)  
- Label `0`: Paper 6 and Paper 1  

---

I wanted to replicate something along the lines of these papers:

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv preprint arXiv:1810.04805.
- Han, C., Wang, Y., Fu, Y., Li, X., Qiu, M., Gao, M., & Zhou, A. (2020). *Meta-learning Siamese Network for Few-Shot Text Classification*. In International Conference on Database Systems for Advanced Applications.

---

I feel like this approach can be successful with some modifications.  
That said, I’m still not sure how to classify those loans, but this is my take on it for now.
While the current model is a basic feedforward classifier built on top of Sentence-BERT embeddings, I believe that a Siamese network architecture may be a better fit for this task. Since the core challenge involves determining whether a fine is likely to be appealed based on textual similarity to known examples, Siamese networks could allow the model to learn better relationships between fines and past appeals. However, to effectively train such a few-shot model, we will need more annotated fine-appeal pairs to provide meaningful contrastive examples during training.


