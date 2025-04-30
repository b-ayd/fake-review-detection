# Fake Review Detection 🕵️‍♂️📝

This project explores the use of Natural Language Processing (NLP) techniques to detect fake (computer-generated) product reviews. It combines traditional TF-IDF vectorization with psycholinguistic features like readability, sentiment consistency, and lexical diversity. The models are evaluated using logistic regression and random forest classifiers.

## 📁 Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) and includes:
- Text reviews
- Star ratings (1–5)
- Category (product domain)
- Label (CG for computer-generated, OR for original)

## ⚙️ Features

- **TF-IDF** (bag-of-words)
- **Readability metrics** (Flesch Reading Ease, FK Grade)
- **Sentiment analysis** (VADER)
- **Sentiment–rating inconsistency**
- **Lexical diversity**

## 🧪 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/b-aydf/fake-review-detection.git
   cd fake-review-detection
   
2. Install dependencies:

pip install -r requirements.txt

3. Run the main script:

python3 main.py
