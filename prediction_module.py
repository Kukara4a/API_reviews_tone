import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import re
import pymorphy2
from gensim.corpora.dictionary import Dictionary
from sklearn.svm import SVC
from gensim.models import LsiModel, TfidfModel

nltk.download('wordnet')
nltk.download('stopwords')


def remove_email(text):
    email = re.compile(r'(?<=\s)\S+\@\S+\.\S+(?=\s)')
    return email.sub(r'', str(text))


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', str(text))


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', str(text))


stop_words = set(stopwords.words("russian"))
morph = pymorphy2.MorphAnalyzer()


def mytokenize(text, stop_words=stop_words):
    text = remove_html(text)
    text = remove_URL(text)
    text = remove_email(text)
    text = text.lower()
    text = word_tokenize(text, language='russian')
    text = [word for word in text if word.isalpha()]
    text = [word for word in text if not word in stop_words]
    text = [morph.normal_forms(word)[0] for word in text]

    return text


def prepare_data(data):
    data['tokens'] = data['Text'].apply(mytokenize)

    return data


def make_vec(X, num_top):
    matrix = np.zeros((len(X), num_top))
    for i, row in enumerate(X):
        matrix[i, list(map(lambda tup: tup[0], row))] = list(map(lambda tup: tup[1], row))
    return matrix


def transform(df, tf_model, model):
    with open('mydict.pkl', 'rb') as file:
        mydict = pickle.load(file)
    corpus = [mydict.doc2bow(text) for text in df]
    corpus = tf_model[corpus]
    corpus = model[corpus]
    corpus = make_vec(corpus, model.num_topics)
    return corpus


# train = pd.read_pickle('train.csv')
# test = pd.DataFrame(pd.read_pickle('test.csv'), columns=['Text'])
#
# train['tokens'] = train['Text'].apply(mytokenize)
# test['tokens'] = test['Text'].apply(mytokenize)
#
# X_all = pd.concat([train.tokens, test.tokens]).reset_index(drop=True)
#
# mydict = Dictionary(X_all)
# corpus = [mydict.doc2bow(text) for text in X_all]
# tf_model = TfidfModel(corpus)
# corpus_tf = tf_model[corpus]
#
# del X_all
#
# lsi_model = LsiModel(corpus_tf, id2word=mydict, num_topics=200)
# X = transform(train["tokens"], tf_model, lsi_model)
# X_test = transform(test["tokens"], tf_model, lsi_model)
#
# model = SVC(C=10, gamma=1, kernel='rbf').fit(X, train['Score'])
#
#
# with open('mydict.pkl', 'wb') as file:
#     pickle.dump(mydict, file)
#
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)
#
# with open('vectorizer.pkl', 'wb') as file:
#     pickle.dump(lsi_model, file)
