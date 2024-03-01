import re
import csv
import nltk
import spacy
import unicodedata
import nltk.tokenize as to

from textblob import TextBlob
from gensim.utils import tokenize
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from keras.preprocessing.text import text_to_word_sequence

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

"""# Normalização"""
def remove_accent(input_str):
  nfkd_form = unicodedata.normalize('NFKD', input_str)
  return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

with open('./dataset/Shakespeare.txt', 'r') as base_file_read, open("ShakespeareNormalized.txt", "w") as base_file_write:
    for line in base_file_read:
        line = line.lower()

        line = remove_accent(line)

        line = re.sub('\.(?!(\S[^. ])|\d)', '', line)
        line = re.sub('(?<!\d)[.,;!?\'#:-](?!\d)', '', line)
        line = re.sub(' +', ' ', line)

        base_file_write.write(line)
        base_file_write.flush()

"""# Tokenização"""
nlp = spacy.load("en_core_web_sm")

tokenizers = {
    "01": lambda x: x.split(),
    "02": nltk.word_tokenize,
    "03": TreebankWordTokenizer().tokenize,
    "04": wordpunct_tokenize,
    "05": TweetTokenizer().tokenize,
    "06": MWETokenizer([('so', 'faithful'), ('may', 'contrive')]).tokenize,
    "07": lambda x: TextBlob(x).words,
    "08": lambda x: [token.text for token in nlp(x)],
    "09": lambda x: list(tokenize(x)),
    "10": text_to_word_sequence
}

for number, tokenizer_func in tokenizers.items():
    with open('ShakespeareNormalized.txt', 'r') as input_file, open(f"Shakespeare_Normalized_Tokenized{number}.txt", "w") as output_file:
        print(f"Making the tokenizer: {tokenizer_func}")
        for line in input_file:
            line_tokens = tokenizer_func(line)
            for item in line_tokens:
                output_file.write(item + '\n')

"""# Stop-words Removal"""
stop_words = stopwords.words('english')

with open('ShakespeareNormalized.txt', 'r') as base_file_read, open("Shakespeare_Normalized_Tokenized_StopWord.txt", "w") as overwritten_base_file:
    for line in base_file_read:
        tokenized_words = nltk.word_tokenize(line)
        line_tokens = [word for word in tokenized_words if not word.lower() in stop_words]
        for item in line_tokens:
            overwritten_base_file.write('\n' + item)
            overwritten_base_file.flush()

"""# Text Lemmatization"""
lemmatizer = WordNetLemmatizer()
with open('Shakespeare.txt', 'r') as base_read_file, open("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt", "w") as overwritten_base_file:
    for line in base_read_file:

        line = line.lower()
        line = remove_accent(line)
        line = re.sub('\.(?!(\S[^. ])|\d)', '', line)
        line = re.sub('(?<!\d)[.,;!?\'#:-](?!\d)', '', line)
        line = re.sub(' +', ' ', line)

        tokenized_words = nltk.word_tokenize(line)
        stop_words = [word for word in tokenized_words if not word.lower() in stop_words]
        line_tokens = [lemmatizer.lemmatize(word) for word in stop_words]
        for item in line_tokens:
            overwritten_base_file.write('\n' + item)
            overwritten_base_file.flush()

"""# Text Stemming"""
stemmers = {
    "01": PorterStemmer(),
    "02": SnowballStemmer("english")
}

for number, stemmer_instance in stemmers.items():
    output_file_path = f'Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming{number}.txt'

    with open('Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt', 'r') as input_file, open(output_file_path, "w") as output_file:
        for line in input_file:
            line_tokens = [stemmer_instance.stem(word) for word in line.split()]
            for item in line_tokens:
                output_file.write(item + '\n')

"""# Análise do Vocabulário"""
lemmatization_file_path = "Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt"
stemming01_file_path = "Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming01.txt"
stemming02_file_path = "Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming02.txt"

lemmatization_csv_path = "Shakespeare_Vocabulary_Lemmatized.csv"
stemming01_csv_path = "Shakespeare_Vocabulary_Porter.csv"
stemming02_csv_path = "Shakespeare_Vocabulary_Snowball.csv"

lem_counter = 0
words_lem_len = 0
words_lem_total = 0

stm_counter_01 = 0
wordstm_len_01 = 0
wordstm_total_01 = 0

stm_counter_02 = 0
wordstm_len_02 = 0
wordstm_total_02 = 0

with open(lemmatization_file_path, "r") as lemmatization_file:
    tokens_lemmatization = [line.strip() for line in lemmatization_file]
    tokens_lemmatization_count = Counter(tokens_lemmatization)
    lem_counter = len(tokens_lemmatization_count)

    for word, freq in tokens_lemmatization_count.items():
        words_lem_len += freq * len(word)
        words_lem_total += freq

with open(stemming01_file_path, "r") as stemming01_file:
    tokens_stemming01 = [line.strip() for line in stemming01_file]
    tokens_stemming01_count = Counter(tokens_stemming01)
    stm_counter_01 = len(tokens_stemming01_count)

    for word, freq in tokens_stemming01_count.items():
        wordstm_len_01 += freq * len(word)
        wordstm_total_01 += freq

with open(stemming02_file_path, "r") as stemming02_file:
    tokens_stemming02 = [line.strip() for line in stemming02_file]
    tokens_stemming02_count = Counter(tokens_stemming02)
    stm_counter_02 = len(tokens_stemming02_count)

    for word, freq in tokens_stemming02_count.items():
        wordstm_len_02 += freq * len(word)
        wordstm_total_02 += freq

mocr_lem = lem_counter / words_lem_total if words_lem_total > 0 else 0
mocr_ste_01 = stm_counter_01 / wordstm_total_01 if wordstm_total_01 > 0 else 0
mocr_ste_02 = stm_counter_02 / wordstm_total_02 if wordstm_total_02 > 0 else 0

m_characters_lem = words_lem_len / words_lem_total
m_characters_ste_01 = wordstm_len_01 / wordstm_total_01
m_characters_ste_02 = wordstm_len_02 / wordstm_total_02

with open(lemmatization_csv_path, "w") as lem_write:
    writer = csv.writer(lem_write)
    writer.writerow(["Token", "Número de ocorrências", "Tamanho em caracteres"])
    for word, freq in tokens_lemmatization_count.items():
        writer.writerow([word, freq, len(word)])

with open(stemming01_csv_path, "w") as stemming_01_write:
    writer_s1 = csv.writer(stemming_01_write)
    writer_s1.writerow(["Token", "Número de ocorrências", "Tamanho em caracteres"])
    for word, freq in tokens_stemming01_count.items():
        writer_s1.writerow([word, freq, len(word)])

with open(stemming02_csv_path, "w") as stemm02_write:
    writer_s2 = csv.writer(stemm02_write)
    writer_s2.writerow(["Token", "Número de ocorrências", "Tamanho em caracteres"])
    for word, freq in tokens_stemming02_count.items():
        writer_s2.writerow([word, freq, len(word)])

with open("Shakespeare_Vocabulary_Analysis.txt", "w") as final_file:
    final_file.write("Word Net Lemmatizer: \nVocabulary size: " + str(lem_counter) + "\nAverage number of occurrences: " + str(mocr_lem) + "\nAverage Token Length: " + str(m_characters_lem))
    final_file.write("\n\nPorter Stemmer: \nVocabulary size: " + str(stm_counter_01) + "\nAverage number of occurrences: " + str(mocr_ste_01) + "\nAverage Token Length: " + str(m_characters_ste_01))
    final_file.write("\n\nSnowball Stemmer: \nVocabulary size: " + str(stm_counter_02) + "\nAverage number of occurrences: " + str(mocr_ste_02) + "\nAverage Token Length: " + str(m_characters_ste_02))