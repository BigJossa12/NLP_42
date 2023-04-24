from nltk.corpus import brown
from nltk.probability import FreqDist
from math import log2
import tqdm

tokens = [word.lower() for word in brown.words()]
freq_dist = FreqDist(tokens)
vocab = set([word for word in tokens if freq_dist[word] >= 10])

w1_w2_pairs = []
for i in range(1, len(tokens)):
    if tokens[i-1] in vocab and tokens[i] in vocab:
        w1_w2_pairs.append((tokens[i-1], tokens[i]))
        
w1_w2_pairs = set(w1_w2_pairs)
print(len(w1_w2_pairs))
pmi_scores = {}
N = len(tokens)
epsilon = 1e-12 

for w1_w2_pair in tqdm.tqdm(w1_w2_pairs):
    if w1_w2_pair not in pmi_scores:
        # Calculate absolute counts
        c_w1_w2 = tokens.count(w1_w2_pair) + epsilon
        c_w1 = freq_dist[w1_w2_pair[0]]
        c_w2 = freq_dist[w1_w2_pair[1]]

        # Calculate PMI
        pmi = log2((c_w1_w2 * N) / (c_w1 * c_w2))
        pmi_scores[w1_w2_pair] = pmi

sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)

with open('pmi.txt','w') as f:
    f.write("Top 20 word pairs based on PMI:\n")
    for pair, pmi in sorted_pmi[:20]:
        f.write(f'{pair}: {pmi}')

    f.write("\nBottom 20 word pairs based on PMI:\n")
    for pair, pmi in sorted_pmi[-20:]:
        f.write(f'{pair}: {pmi}')
