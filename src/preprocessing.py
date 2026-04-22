import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load dataset
data = []
with open('data/News_Category_Dataset_v3.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

print("Total articles:", len(data))

# Take only headlines (simpler)
sentences = [item['headline'] for item in data if item['headline']]

# Limit to 500 sentences (enough)
sentences = sentences[:500]

print("Sample sentence:", sentences[0])

# Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

processed_data = [preprocess(sent) for sent in sentences]

print("Processed sample:", processed_data[0])

# Save processed output
with open('output/processed.txt', 'w') as f:
    for sent in processed_data:
        f.write(" ".join(sent) + "\n")
