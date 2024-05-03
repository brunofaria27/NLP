!pip install -q gdown
!gdown --id 1d-uWMGdXMIu_ptYbNlIaAD1nMtrwXRuK

import re
import nltk
import spacy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.spatial.distance import jaccard
from gensim.models import Word2Vec

def load_data(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return cleaned_text

docs = load_data('headlines.txt')

C = [clean_text(doc) for doc in docs]
C

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def get_tokens(text):
    tokens = re.sub("[^\w]", " ", text).split()
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if not w in ENGLISH_STOP_WORDS]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [ps.stem(w) for w in tokens]
    return tokens

def tokenize(texts):
    words = []
    for text in texts:
        w = get_tokens(text)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def plot_heatmap(matrix, title):
    sns.heatmap(matrix, cmap='coolwarm')
    plt.title(title)
    plt.show()

def calculate_similarity(matrix, metric):
    if metric == "jaccard":
        return 1 - pairwise_distances(matrix, metric=metric)
    elif metric in ["manhattan", "euclidean", "minkowski"]:
        distances = pairwise_distances(matrix, metric=metric)
        return 1 / (1 + distances)
    elif metric == "cosine":
        return sklearn.metrics.pairwise.cosine_similarity(matrix)

V = tokenize(C)

"""# One-Hot Encoding"""
def one_hot_encode(corpus, terms):
    df = pd.DataFrame(columns=terms)
    for i, text in enumerate(corpus):
        words = get_tokens(text)
        bag_vector = np.zeros(len(terms))
        for j, term in enumerate(terms):
            if term in words:
                bag_vector[j] = 1
        df.loc[i] = bag_vector
    return df

one_hot_result = one_hot_encode(C, V)

jaccard_similarity = calculate_similarity(one_hot_result.to_numpy(), "jaccard")
manhattan_similarity = calculate_similarity(one_hot_result, "manhattan")
euclidean_similarity = calculate_similarity(one_hot_result, "euclidean")
minkowski_similarity = calculate_similarity(one_hot_result, "minkowski")
cosine_similarity = calculate_similarity(one_hot_result, "cosine")

plot_heatmap(jaccard_similarity, 'Jaccard')
plot_heatmap(manhattan_similarity, 'Manhattan')
plot_heatmap(euclidean_similarity, 'Euclidean')
plot_heatmap(minkowski_similarity, 'Minkowski')
plot_heatmap(cosine_similarity, 'Cosine')

"""# Count Vectors"""
def remove_repeated_characters(word):
    return re.sub(r'(.)\1+', r'\1\1', word)

def preprocess_corpus(corpus):
    new_corpus = [doc.lower() for doc in corpus]
    punctuation_regex = r"(?<!\d)[\!\?.,;:-](?!\d)"
    new_corpus = [re.sub(punctuation_regex, "", doc, 0) for doc in new_corpus]
    new_corpus = [remove_repeated_characters(word) for word in new_corpus]
    return new_corpus

corpus = preprocess_corpus(C)

vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()
cv = pd.DataFrame(doc_term_matrix.toarray(), columns=terms)

jaccard_cv = calculate_similarity(cv.to_numpy(), metric="jaccard")
manhattan_cv = calculate_similarity(cv, metric="manhattan")
euclidean_cv = calculate_similarity(cv, metric="euclidean")
minkowski_cv = calculate_similarity(cv, metric="minkowski")
cosine_cv = calculate_similarity(cv, metric="cosine")

plot_heatmap(jaccard_cv, 'Jaccard')
plot_heatmap(manhattan_cv, 'Manhattan')
plot_heatmap(euclidean_cv, 'Euclidean')
plot_heatmap(minkowski_cv, 'Minkowski')
plot_heatmap(cosine_cv, 'Cosine')

"""# TF-IDF"""
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(doc_term_matrix)
tf = pd.DataFrame(tfidf_matrix.A, columns=terms)

jaccard_tf = calculate_similarity(tf.to_numpy(), metric="jaccard")
manhattan_tf = calculate_similarity(tf, metric="manhattan")
euclidean_tf = calculate_similarity(tf, metric="euclidean")
minkowski_tf = calculate_similarity(tf, metric="minkowski")
cosine_tf = calculate_similarity(tf, metric="cosine")

plot_heatmap(jaccard_tf, 'Jaccard')
plot_heatmap(manhattan_tf, 'Manhattan')
plot_heatmap(euclidean_tf, 'Euclidean')
plot_heatmap(minkowski_tf, 'Minkowski')
plot_heatmap(cosine_tf, 'Cosine')

"""# n-grams (2-grams)"""
vectorizer = CountVectorizer(ngram_range=(2,2))
doc_term_matrix = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names_out()
ng = pd.DataFrame(doc_term_matrix.A, columns=vocabulary)

jaccard_ng = calculate_similarity(ng.to_numpy(), metric="jaccard")
manhattan_ng = calculate_similarity(ng, metric="manhattan")
euclidean_ng = calculate_similarity(ng, metric="euclidean")
minkowski_ng = calculate_similarity(ng, metric="minkowski")
cosine_ng = calculate_similarity(ng, metric="cosine")

plot_heatmap(jaccard_ng, 'Jaccard')
plot_heatmap(manhattan_ng, 'Manhattan')
plot_heatmap(euclidean_ng, 'Euclidean')
plot_heatmap(minkowski_ng, 'Minkowski')
plot_heatmap(cosine_ng, 'Cosine')

"""# Co-occurrence Vectors (Context Window = 1)"""
def co_occurrence_matrix(doc_term_matrix):
    co_occurrence_matrix = (doc_term_matrix * doc_term_matrix.T)
    g = sp.diags(1. / co_occurrence_matrix.diagonal())
    co_occurrence_matrix_norm = g * co_occurrence_matrix
    return co_occurrence_matrix_norm

def co_occurrence(sentences, window_size):
    co_occurrence_dict = defaultdict(int)
    vocabulary = set()

    for sentence in sentences:
        sentence = sentence.lower().split()
        for i, token in enumerate(sentence):
            vocabulary.add(token)
            next_tokens = sentence[i+1:i+window_size+1]

            for next_token in next_tokens:
                key = tuple(sorted([next_token, token]))
                co_occurrence_dict[key] += 1

    vocabulary = sorted(vocabulary)
    co_occurrence_matrix = pd.DataFrame(data=np.zeros((len(vocabulary), len(vocabulary)), dtype=np.int16),
                                        index=vocabulary,
                                        columns=vocabulary)

    for key, value in co_occurrence_dict.items():
        co_occurrence_matrix.at[key[0], key[1]] = value
        co_occurrence_matrix.at[key[1], key[0]] = value

    return co_occurrence_matrix

co = co_occurrence(corpus, 1)
co.head()

jaccard_co = calculate_similarity(co.to_numpy(), metric="jaccard")
manhattan_co = calculate_similarity(co, metric="manhattan")
euclidean_co = calculate_similarity(co, metric="euclidean")
minkowski_co = calculate_similarity(co, metric="minkowski")
cosine_co = calculate_similarity(co, metric="cosine")

plot_heatmap(jaccard_co, 'Jaccard')
plot_heatmap(manhattan_co, 'Manhattan')
plot_heatmap(euclidean_co, 'Euclidean')
plot_heatmap(minkowski_co, 'Minkowski')
plot_heatmap(cosine_co, 'Cosine')

"""# Word2Vec"""
nlp = spacy.load('en_core_web_sm')
word_2 = [nlp(sentence).vector for sentence in corpus]
word_2_mod = np.array(word_2)

jaccard_wv = calculate_similarity(word_2_mod, metric="jaccard")
manhattan_wv = calculate_similarity(word_2_mod, metric="manhattan")
euclidean_wv = calculate_similarity(word_2_mod, metric="euclidean")
minkowski_wv = calculate_similarity(word_2_mod, metric="minkowski")
cosine_wv = calculate_similarity(word_2_mod, metric="cosine")

plot_heatmap(jaccard_wv, 'Jaccard')
plot_heatmap(manhattan_wv, 'Manhattan')
plot_heatmap(euclidean_wv, 'Euclidean')
plot_heatmap(minkowski_wv, 'Minkowski')
plot_heatmap(cosine_wv, 'Cosine')