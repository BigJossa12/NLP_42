#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

with open("brown_vocab_100.txt.", "r") as file:
    for idx, line in enumerate(file):
        word = line.rstrip()
        word_index_dict[word] = idx

dict_str = str(word_index_dict)

with open("word_to_index_100.txt", "w") as file:
    file.write(dict_str)

print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
