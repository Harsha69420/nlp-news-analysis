import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# LOAD DATASET
# -------------------------

data = []

with open('data/News_Category_Dataset_v3.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# -------------------------
# FILTER 3 CATEGORIES
# -------------------------

categories = ['POLITICS', 'SPORTS', 'TECH']

df = df[df['category'].isin(categories)]

# Use only headline
df = df[['headline', 'category']]

# Drop missing
df = df.dropna()

# Take limited data (faster)
# Balance dataset (equal samples per category)

df_balanced = []

for cat in categories:
    temp = df[df['category'] == cat].sample(n=500, random_state=42)
    df_balanced.append(temp)

df = pd.concat(df_balanced)

print("Balanced dataset:")
print(df['category'].value_counts())

print("Dataset size:", df.shape)

# -------------------------
# TRAIN-TEST SPLIT
# -------------------------

X = df['headline']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -------------------------
# TF-IDF FEATURES
# -------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# NAIVE BAYES MODEL
# -------------------------

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------
# PREDICTIONS
# -------------------------

y_pred = model.predict(X_test_vec)

# -------------------------
# EVALUATION
# -------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))