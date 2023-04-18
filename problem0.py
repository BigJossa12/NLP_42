from nltk.corpus import brown
from nltk import FreqDist
import matplotlib.pyplot as plt

print_answers = True
show_plots = True

# i
f = FreqDist(w for w in brown.words() if any(c.isalpha() for c in w))

# ii
fHumor = FreqDist(w for w in brown.words(categories='humor') if any(c.isalpha() for c in w))
fRomance = FreqDist(w for w in brown.words(categories='romance') if any(c.isalpha() for c in w))


# Number of total tokens, types, and words
n_tokens = len(brown.words())
words = [w for w in brown.words() if any(c.isalpha() for c in w)]
n_words = len(words)
n_types = len(set(words))

# Average number of words per sentence
n_sentences = len(brown.sents())
n_words_in_sentences = sum(len([words for words in sent if any(c.isalpha() for c in words)]) for sent in brown.sents())
avg_words_per_sentence = n_words_in_sentences / n_sentences

# Average word length
total_word_length = sum(len(word) for word in words)
avg_word_length = total_word_length / n_words

# Ten most frequent POS tags
tag_freq = FreqDist(tag for (_, tag) in brown.tagged_words())

if print_answers:
    print("\nNumber of tokens: ", n_tokens)
    print("Number of words: ", n_words)
    print("Number of types: ", n_types)
    print("Average number of words per sentence: ", avg_words_per_sentence)
    print("Average word length: ", avg_word_length)
    print("\nTen most frequent POS tags: ", tag_freq.most_common(10))


# Plot the frequency curve for the corpus on linear and log-log scales
ranks = range(1, len(f)+1)

plt.figure(figsize=(12, 5))
plt.subplot(121)

freqs = [f[1] for f in f.most_common()]
plt.plot(ranks,freqs)

plt.title('Frequency Curve for Brown Corpus (Linear Scale)')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xticks(range(1,502,100))
plt.xlim(1,501)

plt.subplot(122)

plt.plot(ranks,freqs)

plt.title('Frequency Curve for Brown Corpus (Log-Log Scale)')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
if show_plots: plt.show()

# Plot the frequency curves for the two genres on linear and log-log scales
plt.figure(figsize=(12, 5))
plt.subplot(121)

ranksHumor = range(1, len(fHumor)+1)
freqsHumor = [f[1] for f in fHumor.most_common()]
plt.plot(ranksHumor,freqsHumor)

ranksRomance = range(1, len(fRomance)+1)
freqsRomance = [f[1] for f in fRomance.most_common()]
plt.plot(ranksRomance,freqsRomance)

plt.legend(['Humor', 'Romance'])
plt.title('Frequency Curve for Two Genres (Linear Scale)')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xticks(range(1,502,100))
plt.xlim(1,501)

plt.subplot(122)

plt.plot(ranksHumor,freqsHumor)

plt.plot(ranksRomance,freqsRomance)

plt.legend(['Humor', 'Romance'])
plt.title('Frequency Curve for Two Genres (Log-Log Scale)')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
if show_plots: plt.show()