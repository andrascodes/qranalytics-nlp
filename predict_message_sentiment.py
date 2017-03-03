from sklearn.externals import joblib
from textblob import TextBlob
import os

# from message_tokenize import tokenize_only, tokenize_and_stem, get_feature_token


count_vect = joblib.load('./sentiment_analysis_models/count_vect.pkl')
tfidf_transformer = joblib.load('./sentiment_analysis_models/tfidf_transformer.pkl')
clf = joblib.load('./sentiment_analysis_models/clf.pkl')
# count_vect = joblib.load(os.path.join(dir, './sentiment_analysis_models/count_vect.pkl'))
# tfidf_transformer = joblib.load(os.path.join(dir, './sentiment_analysis_models/tfidf_transformer.pkl'))
# clf = joblib.load(os.path.join(dir, './sentiment_analysis_models/clf.pkl'))

def predict_sentiment(message):
    sentiment = TextBlob(message).sentiment.polarity
    if sentiment < -0.33:
        sentiment = 'negative'
    elif sentiment > 0.33:
        sentiment = 'positive'
    else:
        sentiment = 'neutral'

    docs_new = [message]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)[0]

    if sentiment == predicted:
        return predicted
    elif sentiment == 'neutral':
        return predicted
    elif predicted == 'neutral':
        return sentiment
    else:
        return sentiment
