from pathlib import Path
import pickle

import pandas as pd

PACKAGE_PATH = Path(__file__).parent


class MyModel:
    def __init__(self):
        infile = open(PACKAGE_PATH / 'random_forest_model.pickle', 'rb')
        self.rf = pickle.load(infile)
        infile.close()

        v_infile = open(PACKAGE_PATH / 'random_forest_vectorizer.pickle', 'rb')
        self.vectorizer = pickle.load(v_infile)
        v_infile.close()

    def predict(self, input_df):
        input_vectors = self.vectorizer.transform(
            input_df.values.ravel().tolist())
        pred = self.rf.predict_proba(input_vectors)[:, 1]
        pred = pd.DataFrame(pred, columns=['christianity_likelihood'])
        return pred


def get_model():
    return MyModel()
