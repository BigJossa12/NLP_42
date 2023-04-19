#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize

vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i

f = open("brown_100.txt")

counts = np.zeros((len(word_index_dict),len(word_index_dict)))
counts += 0.1

# iterate through file and update counts
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
print(f'p(the | all) = {answer1}')
answer2 = probs[word_index_dict['the']][word_index_dict['jury']]
print(f'p(jury | the) = {answer2}')
answer3 = probs[word_index_dict['the']][word_index_dict['campaign']]
print(f'p(campaign | the) = {answer3}')
answer4 = probs[word_index_dict['anonymous']][word_index_dict['calls']]
print(f'p(calls | anonymous) = {answer4}')

f.close()

with open("smooth_probs.txt", "w") as f:
    f.write(str(answer1) + '\n')
    f.write(str(answer2) + '\n')
    f.write(str(answer3) + '\n')
    f.write(str(answer4) + '\n')