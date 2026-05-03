import nltk
from nltk.tokenize import word_tokenize

# Load some processed sentences
sentences = []

with open('output/processed.txt', 'r') as f:
    for line in f:
        sentences.append(line.strip())

# Take first 30 sentences
sentences = sentences[:30]

for i, sent in enumerate(sentences):
    tokens = word_tokenize(sent)
    pos_tags = nltk.pos_tag(tokens)

    print(f"\nSentence {i+1}: {sent}")
    print("POS Tags:", pos_tags)
