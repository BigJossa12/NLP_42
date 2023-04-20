#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE
import random

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


# Generation of 10 sentences using unigram model -------------------------------------------------------
sentences = []
for i in range(10):
    max_words = random.randint(5,15)
    sentence = GENERATE(word_index_dict=word_index_dict, probs=probs, model_type="unigram", max_words=max_words, start_word="the")
    sentences.append(sentence)
    print(sentence)

with open("unigram_generation.txt", "w") as ug:
    for s in sentences:
        ug.write(s + "\n")


