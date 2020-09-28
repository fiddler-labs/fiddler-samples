import os

from .churn_random_forest import RFPredictor

PACKAGE_DIR = os.path.dirname(__file__)


def get_model():
    return RFPredictor(PACKAGE_DIR, output_column=['probability_churned'])
