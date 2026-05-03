![Python](https://img.shields.io/badge/Python-3.x-blue)

# 📰 NLP-Based News Analysis, Classification and IR System

## 📌 Overview
This project is a comprehensive **Natural Language Processing (NLP)** pipeline designed to process, analyze, and retrieve information from a large dataset of over **200,000 news articles**. It covers the complete workflow from raw text preprocessing to machine learning-based classification and information retrieval.

---

## 🚀 Key Features & Visualizations

### 🔹 1. Text Preprocessing & "News DNA"
- Cleans noisy text (lowercasing, stopword removal, tokenization)
- Converts raw headlines into structured data
- Includes a **Word Cloud visualization** to highlight dominant themes (e.g., COVID, Politics, Sports)

---

### 🔹 2. Language Modeling (N-Grams)
- Implements a **Bigram Language Model**
- Uses **Laplace (Add-1) Smoothing** to handle unseen word pairs
- Demonstrates improvement from infinite perplexity to a measurable score (~1928)

---

### 🔹 3. Machine Learning Classification
- Model: **Naive Bayes**
- Feature Extraction: **TF-IDF Vectorization**
- Dataset balanced across categories:
  - Politics
  - Sports
  - Technology
- Achieved **~84–85% accuracy**
- Includes **Confusion Matrix visualization (Heatmap)**

---

### 🔹 4. Information Retrieval (Search Engine)
- Uses **TF-IDF + Cosine Similarity**
- Accepts user queries and returns most relevant news headlines
- Includes **Relevance Score visualization (Bar Chart)**

---

## 📊 Performance Gallery

| Confusion Matrix | Search Relevance |
|:---:|:---:|
| *(Accuracy ~85% Heatmap)* | *(Bar Chart for Query Results)* |

> Run the scripts to generate these outputs and take screenshots for your report.

---

## 📂 Project Structure

```

nlp-news-analysis/
│
├── data/                # Dataset (JSON file)
├── src/                 # Source code
│   ├── preprocessing.py
│   ├── language_model.py
│   ├── naive_bayes.py
│   ├── ir_system.py
│   ├── pos_tagging.py
│   ├── cfg_parser.py
│   └── ...
│
├── output/              # Generated outputs
├── requirements.txt
└── README.md

````

---

## ▶️ Execution Flow

Run the scripts in this order:

1. **Preprocessing**

   python src/preprocessing.py

* Cleans and prepares text data

2. **Language Model**

   python src/language_model.py

   * Builds bigram model and calculates perplexity

3. **Classification**

   python src/naive_bayes.py

   * Trains Naive Bayes model and outputs accuracy

4. **Information Retrieval**

   python src/ir_system.py
   
   * Runs search engine with ranked results

---

## ⚙️ Setup Instructions

### 1. Clone Repository

git clone https://github.com/Harsha69420/nlp-news-analysis.git
cd nlp-news-analysis

### 2. Create Virtual Environment

python -m venv venv

* Windows:
venv\Scripts\activate

* Mac/Linux:
source venv/bin/activate

---

### 3. Install Dependencies
pip install -r requirements.txt
pip install matplotlib seaborn wordcloud

---

### 4. Dataset

Download **News_Category_Dataset_v3.json** from Kaggle and place it inside the `data/` folder.

---

## 📸 Sample Output

### 🔹 Classification Result

Accuracy: ~0.85

### 🔹 Search Query

Query: cricket match

Top Results:

1. Can Caribbean Cricket Get Its Groove Back?
2. Vatican Cricket Team May Need Some Intervention
3. Kate Middleton Plays Cricket In High Heels

---

## 🧠 Technical Highlights

* **Laplace Smoothing** used to handle zero probabilities in language modeling
* **Balanced dataset** (equal samples per category) for fair classification
* **TF-IDF + Cosine Similarity** used for efficient information retrieval
* End-to-end NLP pipeline implementation

---

## 👨‍💻 Author

**Harsha D.**
*AIML Engineering (3rd Year)*
Project developed for 6th Semester NLP Coursework
