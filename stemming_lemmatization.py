from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Sample words (use domain words)
words = [
    "running", "played", "better", "studies", "cars",
    "winning", "technology", "analysis", "reports", "growing"
]

print(f"{'Word':<15}{'Stemmed':<15}{'Lemmatized':<15}")
print("-" * 45)

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:<15}{stem:<15}{lemma:<15}")