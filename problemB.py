from nltk.corpus import brown
from nltk.probability import FreqDist
from math import log2

# Preprocess corpus
tokens = [word.lower() for word in brown.words()]
freq_dist = FreqDist(tokens)
vocab = set([word for word in tokens if freq_dist[word] >= 10])

# Calculate PMI for all successive word pairs
pmi_scores = {}
N = len(tokens)
epsilon = 1e-12 # small constant to add to probabilities
for i in range(1, len(tokens)):
    if tokens[i-1] in vocab and tokens[i] in vocab:
        w1_w2_pair = (tokens[i-1], tokens[i])
        if w1_w2_pair not in pmi_scores:
            # Calculate joint and individual probabilities
            c_w1_w2 = tokens.count(w1_w2_pair) + epsilon
            c_w1 = tokens.count(tokens[i-1]) + epsilon
            c_w2 = tokens.count(tokens[i]) + epsilon

            # Calculate PMI
            pmi = log2((c_w1_w2 * N) / (c_w1 * c_w2))
            pmi_scores[w1_w2_pair] = pmi

# Print top 20 and bottom 20 word pairs based on PMI
sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
print("Top 20 word pairs based on PMI:\n")
for pair, pmi in sorted_pmi[:20]:
    print(pair[0], pair[1], pmi)

print("\nBottom 20 word pairs based on PMI:\n")
for pair, pmi in sorted_pmi[-20:]:
    print(pair[0], pair[1], pmi)
