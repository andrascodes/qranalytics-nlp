from sklearn.externals import joblib
import json
import pandas as pd
import os

from conversation_tokenize import tokenize_only, tokenize_and_stem, get_feature_token

dir = os.path.dirname(__file__)

# tfidf_vectorizer = joblib.load(os.path.join(dir, './clustering_models/tfidf_vectorizer.pkl'))
# km = joblib.load(os.path.join(dir, './clustering_models/kmeans_model.pkl'))
# vocab_frame = pd.read_pickle(os.path.join(dir, './clustering_models/vocab_frame.pkl'))
tfidf_vectorizer = joblib.load('./clustering_models/tfidf_vectorizer.pkl')
km = joblib.load('./clustering_models/kmeans_model.pkl')
vocab_frame = pd.read_pickle('./clustering_models/vocab_frame.pkl')

# with open('conversations.json', 'r') as f:
#      conversations = json.load(f)


def predict_cluster(new_data):
    new_data = [new_data]
    new_data_frame = pd.DataFrame(tfidf_vectorizer.transform(new_data).toarray())

    prediction = km.predict(new_data_frame)[0]

    word_indexes = km.cluster_centers_[prediction].argsort()[::-1][:6]
    term_names = tfidf_vectorizer.get_feature_names()

    return [prediction, list(map(lambda x: get_feature_token(x, term_names, vocab_frame), word_indexes))]