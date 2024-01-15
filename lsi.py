# Preprocessing file for the data
# Piero Pettenà - January 2024

import os
import pandas as pd
import numpy as np
from collections import Counter
import string

# import nltk
import nltk

nltk.download("wordnet")
nltk.download("punkt")

# preprocessing
from nltk.corpus import stopwords  # stopwords
from nltk import word_tokenize, sent_tokenize  # tokenizing
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
)  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


with open("data/time/TIME.ALL", "r") as f:
    lines = f.read().split("*TEXT ")
    lines = lines[1:]

documents_list = []

for i, article in enumerate(lines):
    parts = article.split("\n\n", 1)
    documents_list.append((parts[0], parts[1].lstrip("\n").lower()))

# Stop words
with open("data/time/TIME.STP", "r") as f:
    sw = f.read().split("\n\n")
stop_words = [word.lower() for word in sw]
del sw


def clean_text(headline):
    le = WordNetLemmatizer()
    word_tokens = word_tokenize(headline)
    tokens = [le.lemmatize(w) for w in word_tokens if w not in stop_words]
    cleaned_text = " ".join(tokens)
    return cleaned_text


data = pd.DataFrame(data=documents_list)
data.columns = ["Article", "Content"]
data.drop(["Article"], axis=1, inplace=True)
data["Cleaned content"] = data["Content"].apply(clean_text)


# Creating corpus removing punctuation from words and excluding stop words
# This creates an array where each line contains a tuple composed of:
#       1. the ID of the document (string)
#       2. the list of words in that document

corpus2 = []
for doc in data["Cleaned content"]:
    list_of_words = [word.strip(string.punctuation) for word in doc.split()]
    corpus2.append(list_of_words)

data["Listed content"] = corpus2
del corpus2
print("Created dataframe 'data'")

# Now it feels like a good time to make a dictionary of all the words in the corpus

# Tokenization and creating a dictionary
words = [word for sentence in data["Listed content"] for word in sentence]
word_counts = Counter(words)
dictionary = {word: idx for idx, (word, _) in enumerate(word_counts.items())}

print("Created dictionary")

pipe = Pipeline(
    [("count", CountVectorizer(vocabulary=dictionary)), ("tfid", TfidfTransformer())]
)
X = pipe.fit_transform(data["Cleaned content"])
print("Fitted pipeline and obtained X:")
print(X.toarray())

print(f"pipe.get_feature_names_out(): {pipe.get_feature_names_out()}")

print("pipe['count'].transform(data['Cleaned content']).toarray()")
print(pipe["count"].transform(data["Cleaned content"]).toarray())
idf = pipe["tfid"].idf_
print(f"pipe['tfid'].idf_ = {idf}")



# vectorizer = CountVectorizer(vocabulary=dictionary)
# X = vectorizer.fit_transform(data["Cleaned content"])
# print(vectorizer.get_feature_names_out())
# print("X:")
# print(X.toarray())
# print(X.shape)

## Now do the tf-idf matrix
# tf_transformer = TfidfTransformer()
# doc_term = tf_transformer.fit_transform(X)
# print(tf_transformer.get_feature_names_out())
# print(tf_transformer.n_features_in_)
# print("doc_term:")
# print(doc_term.toarray())
# idf = tf_transformer.idf_
# print("idf:")
# print(idf)

dd = dict(zip(pipe.get_feature_names_out(), idf))
l = sorted(dd, key=dd.get)


# Latent Semantic Analysis
# documentation states that for LSA n_components should be 100
print("\n********************************************************")
print("Latent Semantic Analysis")
print("********************************************************\n")

lsa_model = TruncatedSVD(
    n_components=10, algorithm="randomized", n_iter=10, random_state=42
)
lsa_top = lsa_model.fit_transform(X)

print(f"lsa_top:\n {lsa_top}")
print(f"lsa_top.shape: {lsa_top.shape}")

# print(f"tf_transformer.feature_names_in: {tf_transformer.feature_names_in_}")
print(
    f"tf_transformer.get_feature_names_out(): {pipe.get_feature_names_out()}"
)

vocab = pipe.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
    print("Topic " + str(i) + ": ")
    for t in sorted_words:
        print(t[0], end=" ")
    print("\n")


# # Example: Transforming text into a list of words, excluding stop words
# def document_to_word_indices(document, dictionary):
#     return [
#         dictionary[word]
#         for word in document
#         if word in dictionary and word not in stop_words
#     ]


# # Transform each document into a list of word indices
# documents_as_word_indices = [
#     document_to_word_indices(doc[1], dictionary) for doc in corpus2
# ]

# corpus = corpus2.copy()
# for i, tpl in enumerate(corpus2):
#     corpus[i] = (tpl[0], documents_as_word_indices[i])

# # Create document-term matrix

# doc_term = np.empty((len(dictionary), len(documents_as_word_indices)))

# for word in dictionary.values():
#     for i, doc in enumerate(corpus):
#         if word in doc[1]:
#             doc_term[word, i] = 1
#         else:
#             doc_term[word, i] = 0
