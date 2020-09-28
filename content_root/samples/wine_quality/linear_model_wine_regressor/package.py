from pathlib import Path

from sklearn_wrapper import SimpleSklearnModel


PACKAGE_PATH = Path(__file__).parent
MODEL_FILE_NAME = 'model.pkl'
PRED_COLUMN_NAMES = ['predicted_quality']


def get_model():
    return SimpleSklearnModel(
        PACKAGE_PATH / MODEL_FILE_NAME, PRED_COLUMN_NAMES)
