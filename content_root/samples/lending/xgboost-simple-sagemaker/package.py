import os

from .sagemaker_xgboost_predictor import SageMakerXGBoostPredictor

PACKAGE_DIR = os.path.dirname(__file__)
SAGE_MAKER_XGB_MODEL_PATH = os.path.join(PACKAGE_DIR, 'xgboost-model')


def get_model():
    return SageMakerXGBoostPredictor(SAGE_MAKER_XGB_MODEL_PATH,
                                     output_column=['loan_status'])
