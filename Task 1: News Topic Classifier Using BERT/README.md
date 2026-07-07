# 📰 News Category Classifier using BERT

## Project Objective

This project fine-tunes a pretrained BERT model to automatically classify news headlines into one of four categories.

- 🌍 World
- 🏆 Sports
- 💼 Business
- 💻 Sci/Tech

The trained model is deployed as an interactive Streamlit web application.

---

## Dataset

AG News Dataset

The dataset contains approximately:

- 120,000 training samples
- 7,600 testing samples

Each sample consists of a news headline and its corresponding category.

---

## Model Used

- BERT (bert-base-uncased)
- Hugging Face Transformers
- Transfer Learning (Fine-tuning)

---

## Libraries

- Transformers
- Datasets
- PyTorch
- Scikit-learn
- Streamlit

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Weighted F1 Score

---

## Features

- Predicts news category from a custom headline
- Displays confidence score
- Interactive Streamlit interface
- Uses a fine-tuned BERT model

---

## Folder Structure on Hugging Face

```
BERT-News-Classifier
│
├── app.py
├── requirements.txt
├── README.md
├── bert_news_classifier/
└── News_Classifier.ipynb
```
---

## Hugging Face App Link
https://huggingface.co/spaces/AJ-381/BERT-News-Classifier

---
---

## Run Locally

```bash
pip install -r requirements.txt

streamlit run app.py
```

---

## Example

Input

```
Apple launches new AI-powered processor
```

Prediction

```
Sci/Tech
```
