# Preprocessing file for the data
# Piero Pettenà - January 2024

import os
import pandas as pd
import numpy as np
from numpy.linalg import norm
from collections import Counter
import string

# import Natural Language Toolkit for preprocessing the data
import nltk

nltk.download("wordnet")
nltk.download("punkt")

from nltk import word_tokenize              # tokenizer
from nltk.stem import WordNetLemmatizer     # lammatizer from WordNet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Importing the data, preprocessing for TIME.ALL format

with open("data/time/TIME.ALL", "r") as f:
    lines = f.read().split("*TEXT ")
    lines = lines[1:]

documents_list = []     # list of tuples of strings (headline, content)

for i, article in enumerate(lines):
    parts = article.split("\n\n", 1)
    documents_list.append((parts[0], parts[1].lstrip("\n").lower()))

# Importing stop words from TIME.STP dataset into a list
with open("data/time/TIME.STP", "r") as f:
    sw = f.read().split("\n\n")
stop_words = [word.lower() for word in sw]

del lines
del sw

# Functions to clean the text: 1) tokenization, 2) lowercasing, 3) lemmatization
# lowercasing is necessary because this function is used for the query as well (not in TIME.ALL format)
# This preprocessing part uses some built-in functions from nltk. The lemmatizer is from WordNet
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
# print(
#     f"Fitted pipeline on 'data['Cleaned content']' and obtained the following tfidf of shape {tfidf.shape}:"
# )
# print(tfidf.toarray())

# print("pipe['count'].transform(data['Cleaned content']).toarray()")
# print(pipe["count"].transform(data["Cleaned content"]))
# print(f"Also learnt the following idf of length {len(pipe['tfid'].idf_)}:")
#print(idf)
idf = pipe["tfid"].idf_ 

dd = dict(zip(pipe.get_feature_names_out(), idf))
sorted_dict = sorted(dd, key=dd.get)

# Latent Semantic Analysis
# documentation states that for LSA n_components should be 100

print("\n********************************************************")
print("Latent Semantic Analysis")
print("********************************************************\n")

lsa_model = TruncatedSVD(
    n_components=10, algorithm="randomized", n_iter=10, random_state=42
)
lsa_matrix = lsa_model.fit_transform(tfidf)

print("Created latent semantic analysis model")
# print(f"lsa_matrix:\n {lsa_matrix}")
# print(f"lsa_matrix.shape: {lsa_matrix.shape}")

# print(f"tf_transformer.feature_names_in: {tf_transformer.feature_names_in_}")
#print(f"tf_transformer.get_feature_names_out(): {pipe.get_feature_names_out()}")

vocab = pipe.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
    # print("Topic " + str(i) + ": ")
    # for t in sorted_words:
    #     print(t[0], end=" ")
    # print("\n")

def get_query():
    print("\n********************************************************")
    query_str = input("Write your free-text query: ")
    # query_str = "In a NumPy array, each row could represent a document, and columns could represent the index and similarity measure."
    query = clean_text(query_str)
    print(f"Nice, what I'm going to use is: {query}")
    return query


def transform_query(query):
    query_vector = pipe.transform([query])
    print(f"query_vector: {query_vector}")
    print(f"query_vector.shape: {query_vector.shape}")
    query_lsa = lsa_model.transform(query_vector).reshape(-1)
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
res_df = pd.DataFrame(data=results)
res_df.columns = ["Similarity measure"]
res_df = res_df.sort_values(by="Similarity measure", ascending=False)


def print_top_results(res_df, n_results=5):
    print(f"\n\nTop 3 results:")
    for i in range(3):
        print(f"{i+1}. {res_df.iloc[i, 0]}")
    print(f"\n\n{res_df.iloc[:n_results]}")
    top3_doc_indices = res_df.iloc[:n_results, 0].index.tolist()
    for doc in top3_doc_indices:
        print(f"\n\n{documents_list[doc][0]}")
        print(f"{documents_list[doc][1]}")

print_top_results(res_df, 3)