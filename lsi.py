# Preprocessing file for the data
# Piero Pettenà - January 2024

import os
import argparse
import math
import pandas as pd
import numpy as np
from numpy.linalg import norm
from collections import Counter
from scipy.sparse import csr_matrix, coo_matrix, find, save_npz, load_npz
import string
import re

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
from sklearn.preprocessing import normalize


# Setting up argparser
parser = argparse.ArgumentParser(
    description="Script that creates an information retrieval system with latent semantic indexing."
)
parser.add_argument(
    "-l",
    "--load_matrices",
    action="store_true",
    help="Load matrices from './matrices/'",
)
parser.add_argument(
    "-s",
    "--sklearn",
    action="store_true",
    help="Use scikit-learn library for doc-ter and tfidf matrix",
)
parser.add_argument(
    "-c",
    "--compare_matrices",
    action="store_true",
    help="""Use both scikit-learn library and custom 
                                                                             methods to create matrices and compare them.""",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Print additional information during run time.",
)
args = parser.parse_args()


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


def remove_thousands_separator(input_string):
    """
    Removes commas used as thousands separators from a string when surrounded by numbers.

    Parameters:
    - input_string (str): The input string with commas as thousands separators.

    Returns:
    - str: The input string with appropriate commas removed.
    """
    result = ""
    for i, char in enumerate(input_string):
        if (
            char == ","
            and i > 0
            and i < len(input_string) - 1
            and input_string[i - 1].isdigit()
            and input_string[i + 1].isdigit()
        ):
            continue  # Skip the comma if it's between two digits
        result += char
    return result


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


def str_to_lst(sentence):
    lst = sentence.split()
    lst = [word.strip(string.punctuation) for word in lst]
    return lst


def str_df_to_lst_df(df):
    df["Listed content"] = df["Cleaned content"].apply(str_to_lst)
    return df


def preprocess_time_data(docs_list):
    df = pd.DataFrame(data=docs_list)
    df.columns = ["Article", "Content"]
    df.drop(["Article"], axis=1, inplace=True)
    df["Cleaned content"] = df["Content"].apply(clean_text)
    df["Cleaned content"] = df["Cleaned content"].apply(remove_thousands_separator)

    return df


data = preprocess_time_data(docs_list=documents_list)
data = str_df_to_lst_df(data)

if args.verbose:
    print("Finished loading data and created dataframe.")

# Now it feels like a good time to make a dictionary of all the words in the corpus

# Tokenization and creating a dictionary
words = [
    word for sentence in data["Listed content"] for word in sentence if len(word) > 0
]
word_counts = Counter(words)
cols_dict = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
if args.verbose:
    print(f"Created dictionary of length {len(cols_dict)}")


def count_words(sentence: str):
    """"Counts the number of appearances of each word in a sentence.
        Returns a dictionary with the words as keys and the number of appearances as values.
    
    Keyword arguments:
    sentence -- string
    """
    counter_dict = {}
    words = re.findall(r"\b[\w-]+\b", sentence)
    for word in words:
        word = word.strip(string.punctuation)   # Remove leading and trailing punctuation
        # Split words containing hyphens
        if "-" in word:
            subwords = word.split("-")
            for subword in subwords:
                counter_dict[subword] = counter_dict.get(subword, 0) + 1
        else:
            counter_dict[word] = counter_dict.get(word, 0) + 1

    return counter_dict


def create_doc_term_matrix(corpus: list, mapper_dictionary: dict):
    """Calculates document (rows) - term (cols) matrix.
    Each cell is the number of appearances. Uses a dictionary,
    so only elements in that dictionary will be counted.
    Returns a sparse matrix.

    Keyword arguments:
    corpus -- list of strings (documents)
    mapper_dictionary -- dict to assign every known word to a column
    """
    nz_tuples = []
    for d_idx, doc in enumerate(corpus):
        counter_dict = count_words(doc)
        for word in counter_dict:
            if word in mapper_dictionary:
                word_index = mapper_dictionary[word]
                nz_tuples.append((d_idx, word_index, counter_dict[word]))
            # else:
            #     if __debug__:
            #         #print(f"{word} excluded because not in mapper dictionary")

    rows, cols, values = zip(*nz_tuples)
    temporary_matrix = coo_matrix((values, (rows, cols)))

    return temporary_matrix.tocsr()  # Convert COO matrix to CSR matrix and return


def get_idf(term_idx: int, count_matrix: csr_matrix):
    """Calculates the idf for a term given its index in the count matrix.
    
    Keyword arguments:
    term_idx -- index of the term in the count matrix
    count_matrix -- Term frequency csr matrix
    """
    n = count_matrix.shape[0]
    # Set of non zero docs for this term
    docs = set(count_matrix[:, term_idx].nonzero()[0])
    df = len(docs)
    tsidfs = []  # List of idf for a term
    for doc in docs:
        # print(f"doc = {doc}")
        tf = count_matrix[doc, term_idx]
        # print(
        #     f"for term {list(cols_dict.keys())[list(cols_dict.values()).index(term_idx)]}, tf = {tf}"
        # )
        idf = math.log((1 + n) / (1 + df), 10) + 1
        # print(f"idf = {idf}")
        # print(f"tf*idf = {tf*idf}")
        tsidfs.append(tf * idf)

    # print(f"tsidfs = {tsidfs}")
    return tsidfs


def calc_tf_idf(count_matrix: csr_matrix):
    """Calculates the tf-idf matrix given a count matrix.
    The aim is to obtain the same results as sklearn.TfidfTransformer
    Returns sparse matrix.

    Keyword arguments
    count_matrix -- Term frequency csr matrix
    """
    if args.verbose:
        print("\nYou've chosen the extremely slow algorithm... this is going to take a while.")
    tuples = []
    n_terms = count_matrix.shape[1]
    for term in range(n_terms):
        if args.verbose:
            if term % 100 == 0:
                print(f"Current term: {term}/{n_terms}")
        idfs = get_idf(term, count_matrix)
        docs = set(count_matrix[:, term].nonzero()[0])
        if len(idfs) == len(docs):
            for doc, idf in zip(docs, idfs):
                tuples.append((doc, term, idf))
        else:
            print("ERROR inside 'calc_tf_idf' -> len(idfs) != len(docs)")
        rows, cols, values = zip(*tuples)
        temporary = coo_matrix((values, (rows, cols)))
        normalized_matrix = normalize(temporary, norm="l2", axis=1)

    return normalized_matrix.tocsr()


def scikit_matr(dataframe: pd.DataFrame, pipeline: Pipeline):
    """Create document/term and tfidf matrix using
    scikit-learn CountVectorizer and TfidfTransformer.
    Returns sparse document/term matrix, tfidf matrix and pipe object.

    Keyword arguments:
    dataframe --    pandas df where column "Listed content" is the one
                    containing the list of words for each document.
    """

    tfidf = pipe.fit_transform([" ".join(doc) for doc in dataframe["Listed content"]])
    doc_t_matrix = pipe["count"].transform(dataframe["Cleaned content"])
    return doc_t_matrix, tfidf


def print_sparse_matrix_difference(matrix1: csr_matrix, matrix2: csr_matrix):
    """Prints differenecs between two sparse matrices.

    Keyword arguments:
    matrix1 -- First sparse matrix
    matrix2 -- Second sparse matrix
    """
    counter = 0
    rows1, cols1, values1 = find(
        matrix1
    )  # Find non-zero elements and their coordinates
    rows2, cols2, values2 = find(matrix2)
    set1 = set(zip(rows1, cols1, values1))
    set2 = set(zip(rows2, cols2, values2))

    num_elements_to_print = 15
    list1 = sorted(set1)
    list2 = sorted(set2)
    if args.verbose:
        print("\tFirst 15 elements for each matrix.")
        for i in range(min(num_elements_to_print, len(list1))):
            print(f"\t\tset1[i] = {list1[i]} \t set2[i] = {list2[i]}")

    differences = set1.symmetric_difference(set2)
    # Find the differences between the two sets
    # Print the differences along with the corresponding values
    for diff in differences:
        row, col, _ = diff
        value1 = matrix1[row, col] if diff in set1 else 0
        value2 = matrix2[row, col] if diff in set2 else 0
        if abs(value1 - value2) >= 0.05:
            counter += 1
    if args.verbose:
        print(f"\tNumber of differences = {counter}\n")


# Creating sklearn pipeline
custom_pattern = r"\b\w+\b|\b\w+-\w+\b"
pipe = Pipeline(
    [
        (
            "count",
            CountVectorizer(vocabulary=cols_dict, token_pattern=custom_pattern),
        ),
        ("tfidf", TfidfTransformer()),
    ]
)


def load_all_matrices():
    """Function to load all sparse matrices from './matrices/'.
    Returns the matrices in the following order:
        doc_t_mtx, true_doc_term_matrix, my_tfidf, tfidf
    """
    if args.load_matrices:
        doc_t_mtx = load_npz("./matrices/my_dt.npz")
        true_doc_t_mtx = load_npz("./matrices/true_dt.npz")
        # TF-IDF matrices
        my_tfidf = load_npz("./matrices/my_tfidf.npz")
        tfidf = load_npz("./matrices/true_tfidf.npz")
        if args.verbose:
            print("\tLoaded term count matrix from './matrices/my_dt.npz'")
            print("\tLoaded true doc-term matrix from './matrices/true_dt.npz'")
            print("\tLoaded tf-idf matrix from './matrices/my_tfidf.npz'")
            print("\tLoaded true tfidf matrix from './matrices/true_tfidf.npz'")

        return doc_t_mtx, true_doc_t_mtx, my_tfidf, tfidf
    else:
        print("ERROR: load_all_matrices shouldn't have been called.")
        return 0


if args.load_matrices:
    print("Set to load matrices mode.")
    doc_t_mtx, true_doc_term_matrix, my_tfidf, tfidf = load_all_matrices()

    if args.compare_matrices:
        print("Set to matrix compare mode.")
        if args.verbose:
            print(
                f"\tCustom document/term matrix number of non-zero elements: {doc_t_mtx.nnz}"
            )
            print(
                f"\tpipe['count'].transform(data['Cleaned content']) number of non-zero elements: {true_doc_term_matrix.nnz}"
            )
            print(
                "\tComparing sklearn document/term sparse matrix (set1) vs custom one (set2)'"
            )
        print_sparse_matrix_difference(true_doc_term_matrix, doc_t_mtx)
        # TF-IDF matrix comparison
        if args.verbose:
            print("Comparing sklearn tf-idf matrix (set1) vs custom one (set2)")
        print_sparse_matrix_difference(tfidf, my_tfidf)
        print("\tNote:\tthis sounds like a terrible number but the calculated")
        print("\t\tvalues and the sklearn ones are actually very close!")

else:
    if args.compare_matrices:
        print("Set to matrix compare mode.")

    if args.sklearn or args.compare_matrices:
        print("Using scikit-learn library.")
        print("Creating document/term matrix...\n")
        true_doc_term_matrix, tfidf = scikit_matr(dataframe=data, pipeline=pipe)
        save_npz("./matrices/true_dt.npz", true_doc_term_matrix)
        print("Saved doc/term matrix to './matrices/true_dt.npz'")
        save_npz("./matrices/true_tfidf.npz", tfidf)

        idf = pipe["tfidf"].idf_
        dd = dict(zip(pipe.get_feature_names_out(), idf))
        sorted_dict = sorted(dd, key=dd.get)

    if not args.sklearn or args.compare_matrices:
        print("Creating document/term matrix...\n")
        doc_t_mtx = create_doc_term_matrix(data["Cleaned content"], cols_dict)
        save_npz("./matrices/my_dt.npz", doc_t_mtx)
        print("Saved matrix to './matrices/my_dt.npz'")

        print("\nUsing custom algorithm to calculate tf-idf")
        my_tfidf = calc_tf_idf(doc_t_mtx)
        save_npz("./matrices/my_tfidf.npz", my_tfidf)
        print("Saved matrix to './matrices/my_tfidf.npz'")

        print(f"tfidf = \n {my_tfidf}")


# **************************************************************************************

# Latent Semantic Analysis
# documentation states that for LSA n_components should be 100

print("\n********************************************************")
print("Latent Semantic Analysis\n")

lsa_model = TruncatedSVD(
    n_components=100, algorithm="randomized", n_iter=10, random_state=42
)

# sklearn's tfidf matrix is still going to be used from now on
_, tfidf = scikit_matr(dataframe=data, pipeline=pipe)


lsa_matrix = lsa_model.fit_transform(tfidf)

print("Created latent semantic analysis model")
pipe.fit([" ".join(doc) for doc in data["Listed content"]])
vocab = pipe.get_feature_names_out()

if args.verbose:
    print(f"\tlsa_matrix:\n\t {lsa_matrix}")
    print(f"\tlsa_matrix.shape: {lsa_matrix.shape}")
    print(f"\ttf_transformer.get_feature_names_out(): {pipe.get_feature_names_out()}")

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]

    if args.verbose:
        print("\tTopic " + str(i) + " - Top 10 most important words: ")
        for t in sorted_words:
            print(f"{t[0]}", end=" ")
        print("\n")


def get_query():
    """Gets query from user and returns it after cleaning it."""
    print("\n********************************************************")
    query_str = input("Write your free-text query: ")
    # query_str = "In a NumPy array, each row could represent a document, and columns could represent the index and similarity measure."
    query = clean_text(query_str)

    if args.verbose:
        print(f"Nice, what I'm going to use is: {query}")
    return query


def transform_query(query):
    """Transforms query into a vector using the pipeline"""
    query_vector = pipe.transform([query])
    print(f"query_vector: {query_vector}")
    print(f"query_vector.shape: {query_vector.shape}")
    query_lsa = lsa_model.transform(query_vector).reshape(-1)
    return query_lsa


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


def print_top_results(res_df, n_results=5):
    print(f"\n\nTop {n_results} results:")
    for i in range(n_results):
        print(f"{i+1}. {res_df.iloc[i, 0]}")
    print("\n\nDoc ID - Similarity measure")
    print(f"{res_df.iloc[:n_results]}")

    if args.verbose:
        topn_doc_indices = res_df.iloc[:n_results, 0].index.tolist()
        for doc in topn_doc_indices:
            print(f"\n\n{documents_list[doc][0]}")
            print(f"{documents_list[doc][1]}")


query = get_query()
query_v = transform_query(query)

results = sim_measures(query_v, lsa_matrix)
res_df = pd.DataFrame(data=results)
res_df.columns = ["Similarity measure"]
res_df = res_df.sort_values(by="Similarity measure", ascending=False)

print_top_results(res_df, 5)