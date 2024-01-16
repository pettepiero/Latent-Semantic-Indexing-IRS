# Preprocessing file for the data
# Piero Pettenà - January 2024

import os
import pandas as pd
import numpy as np
from numpy.linalg import norm
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
    word_tokens = [w.lower() for w in word_tokens]
    tokens = [le.lemmatize(w) for w in word_tokens if w not in stop_words]
    cleaned_text = " ".join(tokens)
    return cleaned_text


data = pd.DataFrame(data=documents_list)
data.columns = ["Article", "Content"]
data.drop(["Article"], axis=1, inplace=True)
data["Cleaned content"] = data["Content"].apply(clean_text)


def str_to_lst(sentence):
    lst = sentence.split()
    lst = [word.strip(string.punctuation) for word in lst]
    return lst


def str_df_to_lst_df(df):
    df["Listed content"] = df["Cleaned content"].apply(str_to_lst)
    return df


data = str_df_to_lst_df(data)

print("Created dataframe 'data'")

# Now it feels like a good time to make a dictionary of all the words in the corpus

# Tokenization and creating a dictionary
words = [word for sentence in data["Listed content"] for word in sentence]
word_counts = Counter(words)
dictionary = {word: idx for idx, (word, _) in enumerate(word_counts.items())}

print(f"Created dictionary of length {len(dictionary)}")

pipe = Pipeline(
    [("count", CountVectorizer(vocabulary=dictionary)), ("tfid", TfidfTransformer())]
)
tfidf = pipe.fit_transform(data["Cleaned content"])
print(
    f"Fitted pipeline on 'data['Cleaned content']' and obtained the following tfidf of shape {tfidf.shape}:"
)
print(tfidf.toarray())

print("pipe['count'].transform(data['Cleaned content']).toarray()")
print(pipe["count"].transform(data["Cleaned content"]).toarray())
print(f"Also learnt the following idf of length {len(pipe['tfid'].idf_)}:")
idf = pipe["tfid"].idf_
print(idf)


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
sorted_dict = sorted(dd, key=dd.get)


# Latent Semantic Analysis
# documentation states that for LSA n_components should be 100

print("\n********************************************************")
print("Latent Semantic Analysis")
print("********************************************************\n")

lsa_model = TruncatedSVD(
    n_components=100, algorithm="randomized", n_iter=10, random_state=42
)
lsa_matrix = lsa_model.fit_transform(tfidf)

print(f"lsa_matrix:\n {lsa_matrix}")
print(f"lsa_matrix.shape: {lsa_matrix.shape}")

# print(f"tf_transformer.feature_names_in: {tf_transformer.feature_names_in_}")
print(f"tf_transformer.get_feature_names_out(): {pipe.get_feature_names_out()}")

vocab = pipe.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
    print("Topic " + str(i) + ": ")
    for t in sorted_words:
        print(t[0], end=" ")
    print("\n")


def get_query():
    print("\n********************************************************")
    # query_str = input("Write your free-text query: ")
    query_str = "In a NumPy array, each row could represent a document, and columns could represent the index and similarity measure."
    query = clean_text(query_str)
    print(f"Your query is: {query}")
    return query


def transform_query(query):
    query_vector = pipe.transform([query])
    print(query_vector.toarray().shape)
    query_lsa = lsa_model.transform(query_vector).reshape(-1)
    print(f"query_lsa = {query_lsa}")
    print(f"query_lsa shape = {query_lsa.shape}")
    return query_lsa


query = get_query()
query_v = transform_query(query)


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def sim_measures(query_vector, docs_matr):
    measures = []
    for doc in docs_matr:
        sim = cos_similarity(query_vector, doc)
        measures.append(sim)
    return measures


def ordered_measures(query_vector, docs_matr):
    measures = sim_measures(query_vector, docs_matr)
    print(f"\n\nMeasures:\n {measures}\n")
    scores = {}
    for i in range(len(measures)):
        scores[i] = measures[i]
    print(f"\n\nscores =")
    print(scores)
    return dict(sorted(scores.items()))


results = sim_measures(query_v, lsa_matrix)
# indices = list(range(len(results)))
# res_idx = list(zip(indices, results))
res_df = pd.DataFrame(data=results)
#res_df = pd.DataFrame(data=res_idx)
res_df.columns = ["Similarity measure"]
# del res_idx
res_df = res_df.sort_values(by="Similarity measure", ascending=False)
print(f"\n\ndataframe: {res_df}")

