import pathlib
import tensorflow as tf
from ..imdb_rnn.package import MyModel

PACKAGE_PATH = pathlib.Path(__file__).parent.parent
SAVED_MODEL_PATH = PACKAGE_PATH / 'imdb_rnn/saved_model'
TOKENIZER_PATH = PACKAGE_PATH / 'imdb_rnn/tokenizer.pickle'


def get_model():
    model = MyModel(
        SAVED_MODEL_PATH,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        TOKENIZER_PATH,
        is_binary_classification=True,
        output_columns=['embedding_input'],
        input_tensor_to_differentiable_layer_mapping=
        {'embedding_input': 'embedding/embedding_lookup:0'})
    model.load_model()
    return model
