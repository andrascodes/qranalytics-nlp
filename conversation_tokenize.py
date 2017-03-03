import re
import nltk
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def tokenize_only(text):
    # first: tokenize by sentence
    # then: tokenize the sentences by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = list(filter(lambda token: re.search('[a-zA-Z]', token), tokens))
    return filtered_tokens

def tokenize_and_stem(text):
    filtered_tokens = tokenize_only(text)
    stems = list(map(stemmer.stem, filtered_tokens))
    return stems

def get_feature_token(ind, terms, vocabulary):
    feature_stem = terms[ind].split(' ')
    token = vocabulary.ix[feature_stem].values.tolist()[0][0]
    return token