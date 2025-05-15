# Spam Detection Using Naive Bayes

A machine learning project to classify SMS messages as spam or ham using a Naive Bayes model.

---

## Project Overview

This project implements a simple yet powerful spam detection system using the **Multinomial Naive Bayes** algorithm, trained on 5,500+ real SMS messages.

---

## Tools and Libraries

| Library | Purpose |
|--------|---------|
| pandas | Data handling |
| sklearn.model_selection | Train-test split |
| sklearn.feature_extraction.text.CountVectorizer | Text vectorization |
| sklearn.naive_bayes.MultinomialNB | Classification model |
| sklearn.metrics | Evaluation metrics |

---

## Dataset

- **File**: `spam.csv`
- **Columns**:
  - `label`: ham or spam
  - `text`: message content

---

## How It Works

1. Load and clean dataset
2. Encode labels (ham → 0, spam → 1)
3. Split into train/test sets
4. Vectorize text using Bag of Words
5. Train Naive Bayes classifier
6. Evaluate using accuracy, confusion matrix & classification report

---

## Results

- **Accuracy**: ~98.38%
- **Confusion Matrix**:

## Author

Developed by [Arunima Surendran](https://github.com/arunimakanavu)

