import pandas as pd
from flask import Flask, request
import pickle
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from prediction_module import prepare_data, transform
import json

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


@app.route('/predict', methods=['POST'])


def predict():
    if request.method == 'POST':
        data = json.loads(request.data)
        data = pd.DataFrame(data, columns=['Text'])
        data = prepare_data(data)

        mydict = Dictionary(data.tokens)
        corpus = [mydict.doc2bow(text) for text in data.tokens]
        tf_model = TfidfModel(corpus)
        vectorized_data = transform(data.tokens, tf_model, vectorizer)

        data['Prediction'] = model.predict(vectorized_data)

        return data[['Text', 'Prediction']].to_json()

if __name__ == '__main__':
    app.run()
