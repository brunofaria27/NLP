from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from matplotlib.pyplot import *
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import nltk
import pandas as pd
import time
import seaborn as sns
import sklearn.mixture as GMM
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import re
import numpy as np
import hdbscan

!pip install hdbscan


plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

all_categories = fetch_20newsgroups(subset='train').target_names
for idx, category in enumerate(all_categories):
    print(f"Index: {idx}, Category: {category}")

selected_category = all_categories[0:1]
newsgroups_train = fetch_20newsgroups(
    subset='train', categories=selected_category)
texts_with_noise = list(newsgroups_train.data)
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
    plt.text(-0.5, 0.6,
             f'Running Time: {end_time - start_time:.2f}s', fontsize=14)
    plt.show()


"""# Vetorização com TF_IDF"""
tfidf_vectorizer = TfidfVectorizer(
    min_df=5, max_df=0.95, stop_words=stop_words_list)
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
plot_clusters(pca_result, hdbscan.HDBSCAN, (), {
              'min_cluster_size': 5, 'min_samples': 5})

"""# Word2Vec"""
tokenized_dataset = [text.split() for text in cleaned_texts]

word2vec_model = Word2Vec(sentences=tokenized_dataset,
                          vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(tokenized_dataset, total_examples=len(
    tokenized_dataset), epochs=100)


def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)


document_vectors = np.array(
    [document_vector(text, word2vec_model) for text in tokenized_dataset])

plot_clusters(pca_result, cluster.KMeans, (), {'n_clusters': 4})
plot_clusters(pca_result, cluster.SpectralClustering, (), {'n_clusters': 6})
plot_clusters(pca_result, GMM.GaussianMixture, (), {'n_components': 6})
plot_clusters(pca_result, AgglomerativeClustering, (), {'n_clusters': 4})
plot_clusters(pca_result, cluster.DBSCAN, (), {'eps': 0.5})
plot_clusters(pca_result, hdbscan.HDBSCAN, (), {
              'min_cluster_size': 5, 'min_samples': 5})

"""# Fazer a classificação"""
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

!pip install - q gdown
!gdown - -id 17WKn4mJkYWg2DSwxnNL74BVHwYOivFso
dataset = pd.read_csv("/content/Tweets.csv")

dataset = dataset.sample(n=1000, random_state=42)
print(dataset.columns)
print(dataset['airline_sentiment'].value_counts())


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
dataset['text'] = dataset['text'].apply(
    lambda x: ' '.join(x) if isinstance(x, list) else x)
print(dataset['text'])

sentiment_mapping = {
    'positive': 1,
    'neutral': 0,
    'negative': 2
}
dataset['airline_sentiment'] = dataset['airline_sentiment'].map(
    sentiment_mapping)

"""# Representação de texto com TF-IDF"""
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(dataset['text']).toarray()
y = dataset['airline_sentiment'].values

print("X shape =", X.shape)
print("y.shape =", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0, shuffle=True)
print(len(x_train))
print(len(x_test))

perform_list = []


def run_model(model_name):
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=100, criterion='entropy', random_state=0)
    elif model_name == 'Multinomial Naive Bayes':
        model = MultinomialNB(alpha=1.0, fit_prior=True)
    elif model_name == 'Support Vector Classifier':
        model = SVC()
    elif model_name == 'Decision Tree Classifier':
        model = DecisionTreeClassifier()
    elif model_name == 'K Nearest Neighbor':
        model = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=4)
    elif model_name == 'Gaussian Naive Bayes':
        model = GaussianNB()

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


models = [
    'Logistic Regression', 'Random Forest', 'Multinomial Naive Bayes',
    'Support Vector Classifier', 'Decision Tree Classifier',
    'K Nearest Neighbor', 'Gaussian Naive Bayes'
]

for model in models:
    run_model(model)

"""# Classificação Naive Bayes"""
classifier = MultinomialNB(alpha=1.0, fit_prior=True).fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
text_to_check = tfidf_vectorizer.transform([text]).toarray()
probabilities = classifier.predict_proba(text_to_check)[0]
max_prob_index = probabilities.argmax()
mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[max_prob_index]
print("\nThe text is classified as:", sentiment)

"""# Decision Tree"""
classifier = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
text_to_check = tfidf_vectorizer.transform([text]).toarray()
predicted_label = classifier.predict(text_to_check)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# SVM"""
classifier = SVC().fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
text_to_check = tfidf_vectorizer.transform([text]).toarray()
predicted_label = classifier.predict(text_to_check)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# Logistic Regression"""
classifier = LogisticRegression().fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
text_to_check = tfidf_vectorizer.transform([text]).toarray()
predicted_label = classifier.predict(text_to_check)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# XGBoost"""
classifier = KNeighborsClassifier(
    n_neighbors=10, metric='minkowski', p=4).fit(x_train, y_train)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
text_to_check = tfidf_vectorizer.transform([text]).toarray()
predicted_label = classifier.predict(text_to_check)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# Random Forest"""
classifier = RandomForestClassifier(
    n_estimators=100, criterion='entropy', random_state=0).fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
text_to_check = tfidf_vectorizer.transform([text]).toarray()
predicted_label = classifier.predict(text_to_check)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# Word2Vec"""
word2vec_model = Word2Vec(sentences=[text.split(
) for text in dataset.text], vector_size=100, window=6, min_count=2, workers=6)


def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)


X = np.array([document_vector(text.split()) for text in dataset.text])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("X shape =", X.shape)
print("y.shape =", y.shape)
print(dataset.head())

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=0, shuffle=True)
print(len(x_train))
print(len(x_test))

perform_list = []

"""# Naive Bayes"""
classifier = MultinomialNB(alpha=1.0, fit_prior=True).fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)
predicted_label = classifier.predict(frase_vector_scaled)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# Decision Tree"""
classifier = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)
predicted_label = classifier.predict(frase_vector_scaled)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# SVM"""
classifier = SVC().fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)
predicted_label = classifier.predict(frase_vector_scaled)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# Logistic Regression"""
classifier = LogisticRegression().fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)
predicted_label = classifier.predict(frase_vector_scaled)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# XGBoost"""
classifier = KNeighborsClassifier(
    n_neighbors=10, metric='minkowski', p=4).fit(x_train, y_train)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)
predicted_label = classifier.predict(frase_vector_scaled)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)

"""# Random Forest"""
classifier = RandomForestClassifier(
    n_estimators=100, criterion='entropy', random_state=0).fit(x_train, y_train)
y_pred = classifier.predict(x_test)

text = "virginamerica really aggressive blast obnoxious entertainment guests faces amp little recourse"
frase_vector = document_vector(text.split()).reshape(1, -1)
frase_vector_scaled = scaler.transform(frase_vector)
predicted_label = classifier.predict(frase_vector_scaled)[0]

mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}
sentiment = mapping[predicted_label]
print("\nThe text is classified as:", sentiment)
