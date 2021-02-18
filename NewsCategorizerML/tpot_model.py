from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer
from copy import copy


def get_model():
    # Average CV score on the training set was: 0.9467387741724025
    return make_pipeline(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        LinearSVC(C=25.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.1)
    )
