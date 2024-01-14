# Preprocessing file for the data
# Piero Pettenà - January 2024

import os
import pandas as pd
import numpy as np
from collections import Counter
import string

with open("data/time/TIME.ALL", "r") as f:
    lines = f.read().split("*TEXT ")
    lines = lines[1:]

documents_list = []

for i, article in enumerate(lines):
    parts = article.split("\n\n", 1)
    documents_list.append((parts[0], parts[1].lstrip("\n")))


# Stop words
with open("data/time/TIME.STP", "r") as f:
    sw = f.read().split("\n\n")
stop_words = [word.lower() for word in sw]
del sw

# documents_list = [word for word in ]

# Creating corpus removing punctuation from words and excluding stop words
# This creates an array where each line contains a tuple composed of:
#       1. the ID of the document (string)
#       2. the list of words in that document
corpus2 = []
for docID, doc in documents_list:
    list_of_words = [
        word.strip(string.punctuation)
        for word in doc.lower().split()
        if word not in stop_words
    ]
    corpus2.append((docID, list_of_words))


# Now it feels like a good time to make a dictionary of all the words in the corpus

# Tokenization and creating a dictionary
words = [word for sentence in corpus2 for word in sentence[1]]
word_counts = Counter(words)
dictionary = {word: idx for idx, (word, _) in enumerate(word_counts.items())}


# Example: Transforming text into a list of words, excluding stop words
def document_to_word_indices(document, dictionary):
    return [
        dictionary[word]
        for word in document
        if word in dictionary and word not in stop_words
    ]


# Transform each document into a list of word indices
documents_as_word_indices = [
    document_to_word_indices(doc[1], dictionary) for doc in corpus2
]

corpus = corpus2.copy()
for i, tpl in enumerate(corpus2):
    corpus[i] = (tpl[0], documents_as_word_indices[i])

# Create document-term matrix

doc_term = np.empty((len(dictionary), len(documents_as_word_indices)))

for word in dictionary.values():
    for i, doc in enumerate(corpus):
        if word in doc[1]:
            doc_term[word, i] = 1
        else:
            doc_term[word, i] = 0

print(doc_term)
print(doc_term.shape)
print(f"First row has {sum(doc_term[0])} matches")