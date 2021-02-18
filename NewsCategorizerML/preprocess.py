from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from scipy import sparse

nlp = spacy.load('en_core_web_trf')
counter = 0


def spacy_preprocess_one(x):
    global counter
    counter += 1
    print('Spacy total parsed:', counter)

    doc = nlp(x)
    x = [token.lemma_.lower() if not token.ent_type_ else token.ent_type_
         for token in doc
         if not token.is_stop and not token.text.isspace()]
    return " ".join(x)


def spacy_preprocess(x):
    return x.apply(spacy_preprocess_one)


# def get_preprocess_pipeline():
#     return Pipeline([
#         ('spacy', FunctionTransformer(spacy_preprocess)),
#         ('tfidf', TfidfVectorizer(lowercase=False,
#                                   ngram_range=(1, 3),
#                                   max_df=0.8,
#                                   min_df=0.01))
#     ])
#
#
# def preprocess_and_store(filename, data):
#     pipe = get_preprocess_pipeline()
#     data = pipe.fit_transform(data)
#     sparse.save_npz(filename, data)
#     return data
#
#
# def load_preprocess(filename):
#     data = sparse.load_npz(filename)
#     return data
