import os

from .tensor_flow_predictor import TensorFlowPredictor

PACKAGE_DIR = os.path.dirname(__file__)
DNN_MODEL_DIR = os.path.join(PACKAGE_DIR, 'dnn_model')

def get_model():
    """ This function is called by the Fiddler executor to instantiate a model predictor.
        Fiddler invokes `predict()` function on this object to run predictions.
    """
    return Model()

class Model():
    """ This implements `predict()` api inovked by Fiddler executor.
        It is a wrapper around TensorFlowPredictor. The TF model
        is located at DNN_MODEL_DIR and output column is 'quality'.
    """

    def predict(self, input_df):
        return self.tf_predictor.predict(input_df)

    def __init__(self):
        self.tf_predictor = TensorFlowPredictor(DNN_MODEL_DIR,
                                                ['predicted_quality'])
