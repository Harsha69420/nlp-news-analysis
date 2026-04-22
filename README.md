# 📰 NLP-Based News Analysis, Classification and Information Retrieval System

## 📌 Overview
This project demonstrates an end-to-end **Natural Language Processing (NLP)** pipeline applied to real-world news data. It includes text preprocessing, language modeling, classification using machine learning, and an information retrieval (search) system.

The system is designed to:
- Understand news text
- Classify headlines into categories
- Retrieve relevant news based on user queries

---

## 🚀 Features

### 🔹 1. Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Cleaning noisy text

---

### 🔹 2. Language Modeling
- Bigram model implementation
- Probability calculation using MLE
- Laplace smoothing
- Perplexity evaluation

---

### 🔹 3. Text Analysis
- Regex for pattern extraction (numbers, dates, capital words)
- Stemming vs Lemmatization comparison
- Part-of-Speech (POS) tagging
- Context-Free Grammar (CFG) parsing

---

### 🔹 4. Text Classification
- TF-IDF vectorization
- Naive Bayes classifier
- Balanced dataset handling
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

### 🔹 5. Information Retrieval System
- TF-IDF based document representation
- Cosine similarity for ranking
- Query-based search
- Top-K relevant results

---

## 🧠 Technologies Used

- Python
- NLTK
- Scikit-learn
- Pandas
- VS Code

---

## 📥 Dataset

The dataset can be downloaded from Kaggle:
News_Category_Dataset_v3.json

---

## 📂 Project Structure
NLP_Capstone_News/
│
├── data/ # Dataset (JSON file)
├── src/ # Source code
│ ├── preprocessing.py
│ ├── language_model.py
│ ├── naive_bayes.py
│ ├── ir_system.py
│ ├── pos_tagging.py
│ └── cfg_parser.py
│
├── output/ # Processed outputs
└── README.md


---


## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/Harsha69420/your-repo-name.git

cd your-repo-name


---

### 2. Create virtual environment (recommended)

#### On Windows:

python -m venv venv
venv\Scripts\activate


#### On Mac/Linux:

python3 -m venv venv
source venv/bin/activate


---

### 3. Install dependencies

pip install -r requirements.txt


---

### 4. Download NLTK resources
Run Python and execute:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


---

### 5. Run the project

#### Preprocessing

python src/preprocessing.py


#### Language Model

python src/language_model.py


#### Naive Bayes Classification

python src/naive_bayes.py


#### Information Retrieval System

python src/ir_system.py

---

## 📊 Sample Results

- Classification Accuracy: ~84%
- Balanced performance across categories
- Relevant search results for user queries

---

## 🧠 Key Concepts

- TF-IDF (Text to numerical conversion)
- Naive Bayes Classification
- Cosine Similarity
- Language Modeling
- NLP Pipeline Design

---

## 📈 Future Improvements

- Use deep learning models (LSTM, BERT)
- Build a web-based interface
- Improve semantic search capabilities

---

## 🎯 Conclusion

This project successfully demonstrates how NLP techniques can be applied to:
- Analyze text
- Classify information
- Build a functional search system

---

## 👨‍💻 Author

- Harsha D.
- AIML Engineering (3rd Year)

---

## ⭐ If you like this project, give it a star!