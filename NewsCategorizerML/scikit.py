from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf():
    return TfidfVectorizer(lowercase=False,
                           ngram_range=(1, 3),
                           max_df=0.5,
                           min_df=0)


def dump_clf(filename, clf):
    dump(clf, filename)


def load_clf(filename):
    return load(filename)
