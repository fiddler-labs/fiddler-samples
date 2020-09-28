import sys

from pathlib import Path

from sklearn_wrapper import SimpleSklearnModel

# on import we have to import and add FeatureTransformer to the __main__
# module because it is expected to be there in the unpickling of the data
# transformer
from .shared_assets.cutsom_feature_transformer import CustomFeatureTransformer
sys.modules['__main__'].CustomFeatureTransformer = CustomFeatureTransformer

PACKAGE_PATH = Path(__file__).parent
MODEL_FILE_NAME = 'lending-club-logreg-model.pkl'
TRANSFORMER_FILE_NAME = ('./shared_assets/'
                         'lending-club-logreg-model-transformer.pkl')
PRED_COLUMN_NAMES = ['probability_charged_off']


def get_model():
    return SimpleSklearnModel(
        PACKAGE_PATH / MODEL_FILE_NAME, PRED_COLUMN_NAMES,
        path_to_serialized_transformer=PACKAGE_PATH / TRANSFORMER_FILE_NAME,
        is_classifier=True)
