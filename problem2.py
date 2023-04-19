#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE



#load the indices dictionary
word_index_dict = {}
with open("brown_vocab_100.txt.", "r") as file:
    for idx, line in enumerate(file):
        word = line.rstrip()
        word_index_dict[word] = idx

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict))
for sentence in f:
    sentence = sentence.lower()
    sentence = sentence.split()
    for word in sentence:
        counts[word_index_dict[word]] += 1
f.close()

print(counts)

# Modified in such a way that all the counts in probs sum up to 1
probs = counts / np.sum(counts)
with open("unigram_probs.txt", "w") as f:
    f.write(str(probs))





