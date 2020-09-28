import pathlib
import tensorflow as tf
from .tf_bert_ig import TFBertModelIg

PACKAGE_PATH = pathlib.Path(__file__).parent
SAVED_MODEL_PATH = PACKAGE_PATH / 'saved_model'


def get_model():
    model = TFBertModelIg(
        SAVED_MODEL_PATH,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        output_columns=['probabilities'],
        is_binary_classification=True,
        output_key='probabilities',
        batch_size=8,
        # There is a trade-off between latency and approximation error of
        # the Integrated Gradients (IG) computation. 'max_allowed_error' is the
        # maximum tolerable approximation error in percentage. Increasing
        # it will improve latency of IG computation but will worsen the
        # approximation error (and therefore faithfulness of the attributions).
        max_allowed_error=5,
        word_level_attribution=True)
    model.load_model()
    return model
