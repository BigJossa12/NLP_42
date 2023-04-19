import numpy as np
from sklearn.preprocessing import normalize


#load the indices dictionary
with open("brown_vocab_100.txt") as vocab:
    word_index_dict = {}
    for i, line in enumerate(vocab):
        word = line.rstrip()
        word_index_dict[word] = i

def trigram_prob(target, evidence):
    string = evidence + target
    total = 0  
    count = 0 

    with open('brown_100.txt') as f:
        for sentence in f:
            words = sentence.lower().split()

            for i in range(len(words)-2):
                if words[i:i+2] == string[:2]: total += 1
                if words[i:i+3] == string: count += 1
    
    return count/total, (count+0.1)/(total+0.1)

print(trigram_prob(['past'],['in', 'the']))
print(trigram_prob(['time'],['in', 'the']))
print(trigram_prob(['said'],['the', 'jury']))
print(trigram_prob(['recommended'],['the', 'jury']))
print(trigram_prob(['that'],['jury', 'said']))
print(trigram_prob([','],['agriculture', 'teacher']))
