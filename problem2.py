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
with open("brown_vocab_100.txt", "r") as file:
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

# print(counts)

# Modified in such a way that all the counts in probs sum up to 1
probs = counts / np.sum(counts)
with open("unigram_probs.txt", "w") as f:
    f.write(str(probs))

# 6. ---------------------------------------------------------------------
# Calculate joint probabilities for the two sentences
sequential_joint_probs = []
with open("toy_corpus.txt", "r") as tc:
    for line in tc:
        line = line.lower()
        line = line.split()
        joint_prob = 1
        for word in line:
            index = word_index_dict[word]
            prob = probs[index]
            joint_prob *= prob
        sequential_joint_probs.append(joint_prob)

# Write probabilities to unigram_eval.txt
with open("unigram_eval.txt", "w") as ue:
    for prob in sequential_joint_probs:
        ue.write(str(prob) + "\n")

# Calculate sentence lengths
sent_lens = []
with open("toy_corpus.txt", "r") as tc:
    for line in tc:
        line = line.lower()
        line = line.split()
        sent_lens.append(len(line))

# Calculate perplexity for both sentences
perplexity = []
with open("unigram_eval.txt", "r") as ue:
    for idx, prob in enumerate(ue):
        prob = float(prob[:-1])
        perplexity.append(1/pow(prob, 1.0/sent_lens[idx]))

# Rewrite unigram_eval.txt with perplexities instead of joint probabilities
with open("unigram_eval.txt", "w") as ue:
    ue.write(str(perplexity) + "\n")



# 7. Generation of 10 sentences using unigram model -------------------------------------------------------
sentences = []
for i in range(10):
    max_words = random.randint(5,15)
    sentence = GENERATE(word_index_dict, probs, "unigram", max_words, "<s>")
    sentences.append(sentence)
    print(sentence)

with open("unigram_generation.txt", "w") as ug:
    for s in sentences:
        ug.write(s + "\n")



