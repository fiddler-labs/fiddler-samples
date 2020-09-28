import pathlib
import pickle
import sys

import pandas as pd
import yaml


class Model:
    THIS_DIR = pathlib.Path(__file__).parent
    MODEL_FILEPATH = THIS_DIR / 'model.pkl'
    MODEL_YAML = THIS_DIR / 'model.yaml'

    def __init__(self):
        self.model = None
        with self.MODEL_YAML.open('r') as yaml_file:
            model_info = yaml.load(yaml_file, Loader=yaml.FullLoader)['model']
        self.task = model_info['model-task']
        self.pred_column_names = [output['column-name']
                                  for output in model_info['outputs']]

    def load_model(self):
        sys.path.append(str(self.THIS_DIR))
        with self.MODEL_FILEPATH.open('rb') as serialized_model:
            self.model = pickle.load(serialized_model)
        sys.path.remove(str(self.THIS_DIR))

    def transform_input(self, input_df):
        return input_df

    def predict(self, input_df):
        if self.task == 'multiclass_classification':
            predict_fn = self.model.predict_proba
        elif self.task == 'binary_classification':
            def predict_fn(x):
                return self.model.predict_proba(x)[:, 1]
        else:
            predict_fn = self.model.predict
        return pd.DataFrame(predict_fn(input_df),
                            columns=self.pred_column_names)


def get_model_class():
    return Model()
