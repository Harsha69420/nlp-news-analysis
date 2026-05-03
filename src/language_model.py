import math
from collections import defaultdict

# -------------------------
# LOAD PROCESSED DATA
# -------------------------

sentences = []
with open('output/processed.txt', 'r') as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)

print("Total sentences:", len(sentences))


# -------------------------
# TRAIN-TEST SPLIT
# -------------------------

split = int(0.8 * len(sentences))
train_data = sentences[:split]
test_data = sentences[split:]

print("Train size:", len(train_data))
print("Test size:", len(test_data))


# -------------------------
# UNIGRAM + BIGRAM COUNTS
# -------------------------

unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in train_data:
    for i in range(len(sentence)):
        unigram_counts[sentence[i]] += 1

        if i < len(sentence) - 1:
            bigram = (sentence[i], sentence[i+1])
            bigram_counts[bigram] += 1

print("Sample unigram:", list(unigram_counts.items())[:5])
print("Sample bigram:", list(bigram_counts.items())[:5])


# -------------------------
# BIGRAM PROBABILITY (MLE)
# -------------------------

def bigram_prob(w1, w2):
    if unigram_counts[w1] == 0:
        return 0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


# -------------------------
# ADD-ONE (LAPLACE) SMOOTHING
# -------------------------

vocab = set(unigram_counts.keys())
V = len(vocab)

def bigram_prob_smooth(w1, w2):
    return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)


# -------------------------
# PERPLEXITY FUNCTION
# -------------------------

def compute_perplexity(test_sentences, smooth=False):
    log_prob = 0
    N = 0

    for sentence in test_sentences:
        for i in range(len(sentence) - 1):
            w1, w2 = sentence[i], sentence[i+1]

            if smooth:
                prob = bigram_prob_smooth(w1, w2)
            else:
                if bigram_counts[(w1, w2)] == 0:
                    return float('inf')
                prob = bigram_prob(w1, w2)

            log_prob += math.log(prob)
            N += 1

    return math.exp(-log_prob / N)


# -------------------------
# PERPLEXITY RESULTS
# -------------------------

pp_no_smooth = compute_perplexity(test_data, smooth=False)
pp_smooth = compute_perplexity(test_data, smooth=True)

print("\nPerplexity WITHOUT smoothing:", pp_no_smooth)
print("Perplexity WITH smoothing:", pp_smooth)
