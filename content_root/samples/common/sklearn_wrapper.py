import pickle

import pandas as pd


class IdentityTransform:
    """Simple placeholder if no transformer is passed in."""
    @staticmethod
    def transform(input_df):
        return input_df


class SimpleSklearnModel:
    """Simple class to deserialize and run a sklearn model on json input."""
    def __init__(self, path_to_serialized_model, output_columns,
                 path_to_serialized_transformer=None, is_classifier=False,
                 is_multiclass=False):
        self.path_to_serialized_model = path_to_serialized_model
        self.path_to_serialized_transformer = path_to_serialized_transformer
        self.output_columns = output_columns
        self.is_cls = is_classifier
        self.is_multi = is_multiclass
        with open(self.path_to_serialized_model, 'rb') as infile:
            self.model = pickle.load(infile)
        if self.path_to_serialized_transformer is not None:
            with open(self.path_to_serialized_transformer, 'rb') as infile:
                self.transformer = pickle.load(infile)
        else:
            self.transformer = IdentityTransform()

    def predict(self, input_df):
        if self.is_cls:
            if self.is_multi:
                predict_fn = self.model.predict_proba
            else:
                def predict_fn(x):
                    return self.model.predict_proba(x)[:, 1]
        else:
            predict_fn = self.model.predict

        return pd.DataFrame(
            predict_fn(self.transformer.transform(input_df)),
            columns=self.output_columns)
