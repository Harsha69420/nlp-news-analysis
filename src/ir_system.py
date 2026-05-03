import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# LOAD DATASET
# -------------------------

data = []

with open('data/News_Category_Dataset_v3.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Use only headlines
df = df[['headline']].dropna()

# Take smaller subset (fast)
# Prefer SPORTS headlines (better search results)

sports_df = df[df['headline'].str.contains('cricket|match|football|sports', case=False, na=False)]

# Combine with some general data
general_df = df.sample(n=700, random_state=42)

df = pd.concat([sports_df, general_df]).drop_duplicates().head(1000)


print("Total documents:", len(df))

documents = df['headline'].tolist()

# -------------------------
# TF-IDF VECTORIZATION
# -------------------------

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    min_df=1
)
doc_vectors = vectorizer.fit_transform(documents)

# -------------------------
# SEARCH FUNCTION
# -------------------------

def search(query, top_k=5):
    query_vec = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    
    # Get sorted indices
    sorted_indices = similarities.argsort()[::-1]

    results = []
    for i in sorted_indices:
        if similarities[i] > 0:
            results.append((documents[i], similarities[i]))
        if len(results) == top_k:
            break
    
    return results

# -------------------------
# TEST SEARCH
# -------------------------

query = "cricket match"
results = search(query)

print("\nQuery:", query)
print("\nTop Results:\n")

for i, (doc, score) in enumerate(results):
    print(f"{i+1}. {doc} (Score: {score:.4f})")




import matplotlib.pyplot as plt

# Your query results
headlines = [
    "Can Caribbean Cricket Get Its Groove Back?",
    "Vatican Cricket Team",
    "Kate Middleton Plays Cricket",
    "Bloody Cricket Bat Found",
    "Jimmy Fallon Eats A Cricket"
]
scores = [0.2969, 0.2409, 0.2307, 0.2163, 0.2078]

plt.figure(figsize=(10, 6))
plt.barh(headlines[::-1], scores[::-1], color='skyblue')
plt.title('Search Relevance for Query: "cricket match"', fontsize=14)
plt.xlabel('Similarity Score', fontsize=12)
plt.tight_layout()
plt.show()
