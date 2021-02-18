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
