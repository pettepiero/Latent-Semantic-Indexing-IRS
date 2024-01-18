# Preprocessing file for the data
# Piero Pettenà - January 2024

import os
import pandas as pd
import numpy as np
from pygtrie import StringTrie
from numpy.linalg import norm
from collections import Counter
from scipy.sparse import csr_matrix, coo_matrix
import string

# import Natural Language Toolkit for preprocessing the data
import nltk

nltk.download("wordnet")
nltk.download("punkt")

from nltk import word_tokenize  # tokenizer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Importing the data, preprocessing for TIME.ALL format

with open("data/time/TIME.ALL", "r") as f:
    lines = f.read().split("*TEXT ")
    lines = lines[1:]

documents_list = []  # list of tuples of strings (headline, content)

for i, article in enumerate(lines):
    parts = article.split("\n\n", 1)
    documents_list.append((parts[0], parts[1].lstrip("\n").lower()))

# Importing stop words from TIME.STP dataset into a list
with open("data/time/TIME.STP", "r") as f:
    sw = f.read().split("\n\n")
stop_words = [word.lower() for word in sw]

del lines
del sw


def remove_contractions(words: list):
    # List of common English contractions
    contractions = [
        "'s",
        "'re",
        "'ve",
        "'d",
        "'ll",
        "'m",
        "'em",
        "n't",
        "'clock",
        "'tis",
        "'twas",
    ]

    # Remove contractions
    words_without_contractions = [word for word in words if word not in contractions]

    return words_without_contractions


# Functions to clean the text: 1) Removal of punctuation 2) tokenization, 3) lowercasing, 4) lemmatization
# lowercasing is necessary because this function is used for the query as well (not in TIME.ALL format)
# This preprocessing part uses some built-in functions from nltk. The lemmatizer is from WordNet
def clean_text(headline):
    le = WordNetLemmatizer()
    word_tokens = word_tokenize(headline)
    word_tokens = [
        w.lower()
        for w in word_tokens
        if w not in string.punctuation and w not in ["``"]
    ]
    word_tokens = remove_contractions(word_tokens)
    tokens = [
        le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w) > 0
    ]
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

print("Created dataframe 'data'\n")

# Now it feels like a good time to make a dictionary of all the words in the corpus

# Tokenization and creating a dictionary
words = [
    word for sentence in data["Listed content"] for word in sentence if len(word) > 0
]
word_counts = Counter(words)
cols_dict = {word: idx for idx, (word, _) in enumerate(word_counts.items())}

print(f"Created dictionary of length {len(cols_dict)}")
# Alternative dict with values = #counts


# needs a cleaned string as parameter
def count_words(sentence: string):
    counter_dict = {}
    list_of_words = [word for word in sentence.split() if len(word) > 0]
    list_of_words = [word.strip(string.punctuation) for word in list_of_words]
    list_of_words = [word for word in list_of_words if len(word) > 0]
    for word in list_of_words:
        counter_dict[word] = counter_dict.get(word, 0) + 1

    return counter_dict


def create_doc_term_matrix(corpus, mapper_dictionary: dict):
    nz_tuples = []
    for d_idx, doc in enumerate(corpus):
        counter_dict = count_words(doc)
        for word in counter_dict:
            word_index = mapper_dictionary[word]
            nz_tuples.append((d_idx, word_index, counter_dict[word]))

    rows, cols, values = zip(*nz_tuples)
    temp_matrix = coo_matrix((values, (rows, cols)))
    # Convert COO matrix to CSR matrix
    return temp_matrix.tocsr()


print("\n*********************************")
print("Creating doc_term matrix")
doc_t_mtx = create_doc_term_matrix(data["Cleaned content"], cols_dict)

print(doc_t_mtx)


dict_trie = StringTrie(cols_dict)
print("********************************")
print("\nCreated dictionary trie")
# for key, value in dict_trie.items():
#     print(f"{key}: {value}")
print("\n********************************")

custom_pattern = r"\b[\w\.-]+\b"

pipe = Pipeline(
    [
        ("count", CountVectorizer(vocabulary=cols_dict, token_pattern=custom_pattern)),
        ("tfid", TfidfTransformer()),
    ]
)

tfidf = pipe.fit_transform([" ".join(doc) for doc in data["Listed content"]])
# tfidf = pipe.fit_transform(data["Cleaned content"])
# print(
#     f"Fitted pipeline on 'data['Cleaned content']' and obtained the following tfidf of shape {tfidf.shape}:"
# )
# print(tfidf.toarray())

print("pipe['count'].transform(data['Cleaned content'])")
true_doc_term_matrix = pipe["count"].transform(data["Cleaned content"])
print(true_doc_term_matrix)
print(true_doc_term_matrix.toarray().shape)


print(
    f"\npipe['count'].transform(data['Cleaned content']).nnz = {pipe['count'].transform(data['Cleaned content']).nnz}"
)

# print(f"The words in the dictionary are: {pipe['count'].get_feature_names_out()[:20]}")

print("\n\n Now trying manually to obtain the same count matrix\n")


def my_count_vect(vocabulary: StringTrie, corpus: list):
    indices = []
    for d_idx, doc in enumerate(corpus):
        # print(f"Checking document {d_idx}")
        # print(f"type of doc is {type(doc)}")
        # print(f"First elements are {doc[:4]}")
        doc_trie = StringTrie.fromkeys(doc, value=1)
        for token in vocabulary:
            print(f"Checking token {token}")
            if doc_trie.has_key(token):
                indices.append((d_idx, vocabulary[token]))
                # print(f"Apparently doc {d_idx} contains '{token}'")

    print(f"\nlen(indices) = {len(indices)}")


my_count_vect(vocabulary=dict_trie, corpus=data["Listed content"])
print(
    f"\npipe['count'].transform(data['Cleaned content']).nnz = {true_doc_term_matrix.nnz}"
)
print(f"doc_t_mtx.nnz = {doc_t_mtx.nnz}")

# print(f"Also learnt the following idf of length {len(pipe['tfid'].idf_)}:")
# print(idf)
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
# print(f"tf_transformer.get_feature_names_out(): {pipe.get_feature_names_out()}")

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
