#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
import random
from generate import GENERATE

#load the indices dictionary
with open("brown_vocab_100.txt") as vocab:
    word_index_dict = {}
    for i, line in enumerate(vocab):
        word = line.rstrip()
        word_index_dict[word] = i

counts = np.zeros((len(word_index_dict),len(word_index_dict)))
counts += 0.1

# iterate through file and update counts
with open("brown_100.txt") as f:
    for sentence in f:
        sentence = sentence.lower()
        sentence = sentence.split()
        
        previous_word = '<s>'
        for word in sentence[1:]:
            counts[word_index_dict[previous_word]][word_index_dict[word]] += 1
            previous_word = word

# normalize counts
probs = normalize(counts, norm='l1', axis=1)

# writeout bigram probabilities
answer1 = probs[word_index_dict['all']][word_index_dict['the']]
answer2 = probs[word_index_dict['the']][word_index_dict['jury']]
answer3 = probs[word_index_dict['the']][word_index_dict['campaign']]
answer4 = probs[word_index_dict['anonymous']][word_index_dict['calls']]

with open("smooth_probs.txt", "w") as f:
    f.write(str(answer1) + '\n')
    f.write(str(answer2) + '\n')
    f.write(str(answer3) + '\n')
    f.write(str(answer4) + '\n')

# print(counts[word_index_dict['anonymous'],])

# 6. ------------------------------------------------------------------
sequential_joint_probs = []
with open("toy_corpus.txt", "r") as tc:
    for line in tc:
        line = line.lower()
        line = line.split()
        joint_prob = 1
        previous_word = "<s>"
        for idx, word in enumerate(line[1:]):
            prob = probs[word_index_dict[previous_word]][word_index_dict[word]]
            joint_prob *= prob
            previous_word = word
        sequential_joint_probs.append(joint_prob)

# Calculate sentence lengths
sent_lens = []
with open("toy_corpus.txt", "r") as tc:
    for line in tc:
        line = line.lower()
        line = line.split()
        sent_lens.append(len(line))

# Calculate perplexity for both sentences
perplexity = []
for idx, prob in enumerate(sequential_joint_probs):
    perplexity.append(1/pow(prob, 1.0/(sent_lens[idx]-1)))

print(perplexity)
with open("smoothed_eval.txt", "w") as be:
    for p in perplexity:
        be.write(str(p) + "\n")

# 7. Generation of 10 sentences using bigram model -------------------------------------------------------
sentences = []
for i in range(10):
    max_words = random.randint(5,15)
    sentence = GENERATE(word_index_dict, probs, "bigram", max_words, "<s>")
    sentences.append(sentence)
    print(sentence)

with open("smoothed_generation.txt", "w") as ug:
    for s in sentences:
        ug.write(s + "\n")

