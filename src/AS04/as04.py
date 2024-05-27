!pip install hdbscan
import hdbscan

import warnings
warnings.simplefilter("ignore")

import numpy as np
import re
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import json
import sklearn.mixture as GMM

import seaborn as sns
import time
import pandas as pd
from sklearn.decomposition import PCA
import nltk
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
from matplotlib.pyplot import *
from nltk.corpus import stopwords
from matplotlib.pyplot import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Essa parte do código ocorreu, provavelmente o servidor onde pega os dados dos 20newsgroups esta com problema dia ocorrido 26/05
# HTTPError: HTTP Error 403: Forbidden
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

try:
    all_categories = fetch_20newsgroups(subset='train').target_names
    for idx, category in enumerate(all_categories):
        print(f"Index: {idx}, Category: {category}")
except Exception as e:
    print(f"Error fetching the dataset: {e}")
finally:
    # Feito para contornar o problema que estava sendo obtido
    !pip install -q gdown
    !gdown --id 1saWHfZWnitDnHGFgH0JCypXJum-RYsbe
    with open('20newsgroups_train.json', 'r') as infile:
        data = json.load(infile)

    all_categories = data['target_names']
    for idx, category in enumerate(all_categories):
        print(f"Index: {idx}, Category: {category}")

selected_category = all_categories[0:1]
try:
  newsgroups_train = fetch_20newsgroups(subset='train', categories=selected_category)
  texts_with_noise = list(newsgroups_train.data)
except Exception as e:
  print(f"Error fetching the dataset: {e}")
finally:
  with open('20newsgroups_train.json', 'r') as infile:
        data = json.load(infile)
  texts_with_noise = [data['data'][i] for i in range(len(data['target'])) if data['target'][i] == all_categories.index(selected_category[0])]

cleaned_texts = [re.sub(r'[^\w\s]', '', text) for text in texts_with_noise]

stop_words_list = list(stop_words)
print(stop_words_list)

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.figure()  # Create a new figure for each plot
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(f'Clusters by {algorithm.__name__}', fontsize=18)
    plt.text(-0.5, 0.6, f'Running Time: {end_time - start_time:.2f}s', fontsize=14)
    plt.show()

"""# Vetorização com TF_IDF"""
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, stop_words=stop_words_list)
tfidf_values = tfidf_vectorizer.fit_transform(cleaned_texts)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_values.toarray())

plt.scatter(pca_result.T[0], pca_result.T[1], c='b', label='Data points')
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.show()

plot_clusters(pca_result, cluster.KMeans, (), {'n_clusters': 4})
plot_clusters(pca_result, cluster.SpectralClustering, (), {'n_clusters': 6})
plot_clusters(pca_result, GMM.GaussianMixture, (), {'n_components': 6})
plot_clusters(pca_result, AgglomerativeClustering, (), {'n_clusters': 4})
plot_clusters(pca_result, cluster.DBSCAN, (), {'eps': 0.25})
plot_clusters(pca_result, hdbscan.HDBSCAN, (), {'min_cluster_size': 5, 'min_samples': 5})

"""# Word2Vec"""
tokenized_dataset = [text.split() for text in cleaned_texts]

word2vec_model = Word2Vec(sentences=tokenized_dataset, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(tokenized_dataset, total_examples=len(tokenized_dataset), epochs=100)

def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)

document_vectors = np.array([document_vector(text, word2vec_model) for text in tokenized_dataset])

plot_clusters(pca_result, cluster.KMeans, (), {'n_clusters': 4})
plot_clusters(pca_result, cluster.SpectralClustering, (), {'n_clusters': 6})
plot_clusters(pca_result, GMM.GaussianMixture, (), {'n_components': 6})
plot_clusters(pca_result, AgglomerativeClustering, (), {'n_clusters': 4})
plot_clusters(pca_result, cluster.DBSCAN, (), {'eps': 0.5})
plot_clusters(pca_result, hdbscan.HDBSCAN, (), {'min_cluster_size': 5, 'min_samples': 5})

"""# Fazer a classificação"""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

!gdown --id 1tn-5eZLOlAxrP9-4Tzpb693Wfj1L-tti
dataset = pd.read_csv("/content/Tweets.csv")

dataset = dataset.sample(n=1000, random_state=42)

def remove_tags(text):
    return re.sub(r'<.*?>', '', text)

def special_char(text):
    return ''.join([x if x.isalnum() else ' ' for x in text])

def convert_lower(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]

dataset['text'] = dataset['text'].apply(remove_tags)
dataset['text'] = dataset['text'].apply(special_char)
dataset['text'] = dataset['text'].apply(convert_lower)
dataset['text'] = dataset['text'].apply(remove_stopwords)
dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
print(dataset['text'])

sentiment_mapping = {
    'positive': 1,
    'neutral': 0,
    'negative': 2
}

# Usado para testar mais abaixo
mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}

classifiers = {
    'MultinomialNB': MultinomialNB(alpha=1.0, fit_prior=True),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SVC': SVC(),
    'LogisticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=4),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
}

dataset['airline_sentiment'] = dataset['airline_sentiment'].map(sentiment_mapping)

def run_model(model_name, model):
    one_vs_rest = OneVsRestClassifier(model)
    one_vs_rest.fit(x_train, y_train)
    y_pred = one_vs_rest.predict(x_test)

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision, recall, fscore, support = score(y_test, y_pred, average='micro')

    print(f'Test Accuracy Score of Basic {model_name}: {accuracy}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {fscore}')

    perform_list.append({
        'Model': model_name,
        'Test Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1-score': round(fscore, 2)
    })

"""# Representação de texto com TF-IDF"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(dataset['text']).toarray()
y = dataset['airline_sentiment'].values

print("X shape =", X.shape)
print("y.shape =", y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=True)
print(len(x_train))
print(len(x_test))

perform_list = []

for model_name, model in classifiers.items():
    run_model(model_name, model)

"""# Testando classificação"""

text = "Very aggressive employees I didn't like flying with them"
text_to_check = tfidf_vectorizer.transform([text]).toarray()

for name, clf in classifiers.items():
    clf.fit(x_train, y_train)

    if name == 'MultinomialNB':
        probabilities = clf.predict_proba(text_to_check)[0]
        max_prob_index = probabilities.argmax()
        sentiment = mapping[max_prob_index]
    else:
        predicted_label = clf.predict(text_to_check)[0]
        sentiment = mapping[predicted_label]

    print(f"\nThe text is classified as: {sentiment} by {name}")

"""# Word2Vec"""
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

word2vec_model = Word2Vec(sentences=[text.split() for text in dataset.text], vector_size=100, window=6, min_count=2, workers=6)

def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

X = np.array([document_vector(text.split()) for text in dataset.text])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0, shuffle=True)

perform_list = []

for model_name, model in classifiers.items():
    run_model(model_name, model)

"""# Testando classificações"""
text = "Very aggressive employees I didn't like flying with them"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)

for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    predicted_label = clf.predict(frase_vector_scaled)[0]
    sentiment = mapping[predicted_label]
    print(f"\nThe text is classified as: {sentiment} by {name}")