import numpy as np
import pathlib
import pickle
import logging
import pandas as pd
import tensorflow as tf
from .cover_tokens import strip_accents_and_special_characters
from .cover_tokens import word_tokenizer
from .cover_tokens import cover_tokens_new as cover_tokens
from .cover_tokens import regroup_attributions
from .tf_saved_model_wrapper_ig import TFSavedModelWrapperIg

PACKAGE_PATH = pathlib.Path(__file__).parent
SAVED_MODEL_PATH = PACKAGE_PATH / 'saved_model'
TOKENIZER_PATH = PACKAGE_PATH / 'tokenizer.pickle'

LOG = logging.getLogger(__name__)


class MyModel(TFSavedModelWrapperIg):
    def __init__(self, saved_model_path, sig_def_key, tokenizer_path,
                 is_binary_classification=False,
                 output_key=None,
                 batch_size=8,
                 output_columns=[],
                 input_tensor_to_differentiable_layer_mapping={},
                 max_allowed_error=None):
        """
        Class to load and run the IMDB RNN model.
        See: TFSavedModelWrapper

        Args:
        :param saved_model_path: Path to the directory containing the TF
            model in SavedModel format.

        :param sig_def_key: Key for the specific SignatureDef to be used for
            executing the model.

        :param tokenizer_path: Path to tokenizer (pickle file) used to tokenize
            the text input.

        :param is_binary_classification: if the model is a binary
            classification model. If True, the number of output columns is one.

        :param output_key: output_key parameter as specified in the
            TFSavedModelWrapper class.

        :param batch_size: the batch size for input into the model. Depends
            on model and instance config.

        :param output_columns: output_columns parameter as specified in the
            TFSavedModelWrapper class.

        :param max_allowed_error: (int) the absolute value of the maximum
            allowed integral approximation error for the IG computation.
            The error must be expressed as a percentage. If None then IG
            will be calculated for a pre-determined number of steps.
            Otherwise, the number of steps will be increased till
            the error is within the specified limit
        """
        super().__init__(saved_model_path, sig_def_key,
                         is_binary_classification=is_binary_classification,
                         output_key=output_key,
                         batch_size=batch_size,
                         output_columns=output_columns,
                         input_tensor_to_differentiable_layer_mapping=
                         input_tensor_to_differentiable_layer_mapping,
                         max_allowed_error=max_allowed_error)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_seq_length = 512

    def transform_input(self, input_df):
        """
        Transform the provided dataframe into one that complies with the input
        interface of the model.

        Overrides the transform_input method of TFSavedModelWrapper.
        """

        input_tokens = (input_df['sentence']
                        .apply(lambda x: self.tokenizer.encode(
                                strip_accents_and_special_characters(x))))

        input_tokens = input_tokens.apply(lambda x: self._pad(x))

        return pd.DataFrame({'embedding_input': input_tokens.values.tolist()})

    def generate_baseline(self, input_df):

        input_tokens = input_df['sentence'].apply(lambda x:
                                                  self.tokenizer.encode(''))
        input_tokens = input_tokens.apply(lambda x: self._pad(x))

        return pd.DataFrame({'embedding_input': input_tokens.values.tolist()})

    def project_attributions(self, input_df, transformed_input_df,
                             attributions):
        """
        Maps the transformed input to original input space so that the
        attributions correspond to the features of the original input.
        Overrides the project_attributions method of TFSavedModelWrapper.
        """

        wordpiece_tokens = [self.tokenizer.decode([int(t)]) for t in
                            (transformed_input_df['embedding_input'][0]
                             .tolist())]

        word_tokens = word_tokenizer(
            strip_accents_and_special_characters(
                input_df['sentence'].iloc[0]))

        coverings = cover_tokens(word_tokens,
                                 wordpiece_tokens,
                                 num_fine_tokens_to_be_matched=
                                 self.max_seq_length)

        word_attributions = regroup_attributions(
            coverings,
            attributions['embedding_input'][0].astype(
                'float').tolist())
        if word_attributions:
            return {'embedding_input': [word_tokens, word_attributions]}
        else:
            LOG.info('Cover tokens failed.  Falling back to wordpiece tokens')
            return {'embedding_input': [wordpiece_tokens,
                                        attributions['embedding_input'
                                                     ][0].astype(
                                                     'float').tolist()
                                        ]}

    def _pad(self, a):
        a_padded = np.zeros(self.max_seq_length)
        a_padded[:min(len(a), self.max_seq_length)] = \
            a[:min(len(a), self.max_seq_length)]
        return a_padded


def get_model():
    model = MyModel(
        SAVED_MODEL_PATH,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        TOKENIZER_PATH,
        is_binary_classification=True,
        batch_size=200,
        output_columns=['embedding_input'],
        input_tensor_to_differentiable_layer_mapping=
        {'embedding_input': 'embedding/embedding_lookup:0'},
        max_allowed_error=5)
    model.load_model()
    return model
