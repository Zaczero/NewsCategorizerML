import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from scikit import create_tfidf
from preprocess import spacy_preprocess
from tpot_model import get_model

TRAIN_FILE = "data_train.csv"
TEST_FILE = "data_test.csv"

SPACY_TRAIN_FILE = "spacy_data_train.csv"
SPACY_TEST_FILE = "spacy_data_test.csv"

TFIDF_JOBLIB = "tfidf.joblib"
TFIDF_TRAIN_FILE = "tfidf_data_train.npz"
TFIDF_TEST_FILE = "tfidf_data_test.npz"

section_to_name = {
    0: 'Politics',
    1: 'Technology',
    2: 'Entertainment',
    3: 'Business'
}

with open("news.txt", "r", encoding='utf-8') as f:
    news = f.read()


def train_preprocess():
    train_df = pd.read_csv(TRAIN_FILE, header=0, names=['x', 'y'])
    train_df['x'] = spacy_preprocess(train_df.x)

    train_df.to_csv(SPACY_TRAIN_FILE, index=False)


def optimize_model(x, y):
    tpot = TPOTClassifier(generations=100,
                          population_size=20,
                          scoring='accuracy',
                          cv=3,
                          n_jobs=14,
                          config_dict='TPOT sparse',
                          memory='auto',
                          periodic_checkpoint_folder='checkpoints',
                          early_stop=10,
                          verbosity=2)

    tpot.fit(x, y)
    tpot.export('tpot_model.py')


# train_preprocess()

spacy_train_df = pd.read_csv(SPACY_TRAIN_FILE, header=0)
X_train, X_test, y_train, y_test = train_test_split(spacy_train_df.x, spacy_train_df.y,
                                                    test_size=.2,
                                                    random_state=42,
                                                    stratify=spacy_train_df.y)

vect = create_tfidf()

X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)

# optimize_model(X_train, y_train)

model = get_model()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

news_df = pd.DataFrame({'x': [news]})
news_spacy = spacy_preprocess(news_df.x)
news_vect = vect.transform(news_spacy)

print(model.predict(news_vect))
