import re
import numpy as np
import spacy
import pandas as pd
import scipy.sparse as sp

from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import warnings
warnings.filterwarnings("ignore")

def get_tokens(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def tokenize(texts):
    words = []
    for text in texts:
        w = get_tokens(text)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def preprocess_corpus(corpus):
    new_corpus = [re.sub(r"(?<!\d)[\!?.,;:-](?!\d)", "", doc.lower()) for doc in corpus]
    return new_corpus

def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        text = get_tokens(text)
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple(sorted([t, token]))
                d[key] += 1

    vocab = sorted(vocab)
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df

all_categories = fetch_20newsgroups(subset='train').target_names
input_rows = [
    'something! !!.ç is alive in the ocean',
    'océan män',
    'ocean ocean'
]
selected_categories = all_categories[0:1] # Used to run only the first test
# selected_categories = all_categories[18:20]
new_selected = fetch_20newsgroups(subset='train', categories=selected_categories)

# texts = list(new_selected.data) # Select this line if you want more data
texts = input_rows
for text in texts:
   text_words = get_tokens(text)

windex = {word: index for index, word in enumerate(texts)}
chunk = 1000

preprocessed_texts = preprocess_corpus(texts)
co_occurrence_matrix = co_occurrence(preprocessed_texts, window_size=5)
corpus = preprocess_corpus(texts)


"""# One-Hot Encoding"""
total = len(texts)
num_chunks = total // chunk + (total % chunk != 0)

def one_hot_encode(texts, vocabulary):
    with open("20News_01.txt", "w") as outfile:
        for text in texts:
            tokens = get_tokens(text)
            bag_vector = np.zeros(len(vocabulary))
            for token in tokens:
                if token in vocabulary:
                    bag_vector[vocabulary[token]] = 1
            outfile.write("{0} = {1}".format(text, np.array(bag_vector)))
            outfile.write("\n")
            outfile.flush()

vocabulary_index = {word: index for index, word in enumerate(co_occurrence_matrix.index)}
one_hot_encode(texts, vocabulary_index)


"""# Count Vectors"""
def generate_count_vectors(texts, terms):
    total_terms = len(terms)
    num_chunks = total_terms // chunk + (total_terms % chunk != 0)

    for i in range(num_chunks):
        start_idx = i * chunk
        end_idx = min((i + 1) * chunk, total_terms)
        subset = pd.DataFrame(columns=terms[start_idx:end_idx])

        for text in texts:
            words = get_tokens(text)
            bag_vector = np.zeros(end_idx - start_idx)
            for w in words:
                for j, term in enumerate(terms[start_idx:end_idx]):
                    if term == w:
                        bag_vector[j] += 1
            subset = subset.append(pd.Series(bag_vector, index=subset.columns), ignore_index=True)

        with open("20News_02.txt", "a") as f:
            for index, row in subset.iterrows():
                string_row = "\t".join(map(str, row))
                f.write(string_row + "\n")

vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()

generate_count_vectors(texts, terms)


"""# TF-IDF Vectors"""
def get_tf_idf_vectors(doc_term_matrix, terms):
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(doc_term_matrix)
    return pd.DataFrame(tfidf_matrix.A, columns=terms)

def save_tf_idf_vectors_to_file(tfidf_matrix, terms):
    with open("20News_03.txt", "w") as outfile:
        outfile.write(get_tf_idf_vectors(tfidf_matrix, terms).to_string())
        outfile.flush()

save_tf_idf_vectors_to_file(doc_term_matrix, terms)


"""# N-GRAMS"""
def process_ngrams(corpus):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    doc_term_matrix = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names_out()

    df = pd.DataFrame(doc_term_matrix.toarray(), columns=vocabulary)

    total_cols = df.shape[1]
    num_chunks = total_cols // chunk + (total_cols % chunk != 0)

    for i in range(num_chunks):
        start = i * chunk
        end = start + chunk
        subset = df.iloc[:, start:end]
        with open("20News_04.txt", "a") as file:
            for index, row in subset.iterrows():
                string_row = "\t".join(map(str, row))
                file.write(string_row + "\n")

process_ngrams(preprocessed_texts)


"""# Co-Occurrence Vectors"""
def calculate_co_occurrence_vectors(doc_term_matrix):
    co_occurrence_matrix = (doc_term_matrix.T * doc_term_matrix)
    g = sp.diags(1. / co_occurrence_matrix.diagonal())
    co_occurrence_matrix_norm = g * co_occurrence_matrix
    return co_occurrence_matrix_norm

def write_co_occurrence_chunks(df):
    total_cols = df.shape[1]
    num_chunks = total_cols // chunk + (total_cols % chunk != 0)

    for i in range(num_chunks):
        start_idx = i * chunk
        end_idx = start_idx + chunk
        subset = df.iloc[:, start_idx:end_idx]
        with open("20News_05.txt", "a") as f:
            for _, row in subset.iterrows():
                string_row = "\t".join(map(str, row))
                f.write(string_row + "\n")

co_occurrence_matrix_norm = calculate_co_occurrence_vectors(doc_term_matrix)
co_occurrence_df = co_occurrence(corpus, 1)
write_co_occurrence_chunks(co_occurrence_df)


"""# Word2Vec"""
def word2vec_to_file(corpus):
    nlp = spacy.load('en_core_web_sm')

    with open("20News_06.txt", "a") as file:
        for sentence in corpus:
            vec = nlp(sentence).vector
            vec_str = ' '.join(map(str, vec))
            file.write(f"{vec_str}\n")

word2vec_to_file(preprocessed_texts)