import logging
import numpy as np
import pathlib
import pickle
from bert import tokenization
from .tf_saved_model_wrapper_ig import TFSavedModelWrapperIg
from .cover_tokens import cover_tokens

PACKAGE_PATH = pathlib.Path(__file__).parent
BERT_BASE_TOKENIZER_PATH = PACKAGE_PATH / 'bert_tokenizer.pkl'


class TFBertModelIg(TFSavedModelWrapperIg):
    def __init__(self,
                 saved_model_path,
                 sig_def_key,
                 output_columns,
                 is_binary_classification=False,
                 output_key=None,
                 batch_size=8,
                 input_tensor_to_differentiable_layer_mapping={
                     'input_ids':
                         'module_apply_tokens/bert/embeddings/add_1:0'},
                 max_seq_length=256,
                 tokenizer_path=BERT_BASE_TOKENIZER_PATH,
                 max_allowed_error=None,
                 word_level_attribution=True):
        """
        This class offers methods to load and run Integrated Gradients (IG)
        explanations on a fine-tuned BERT model.

        See: https://github.com/google-research/bert
        See: https://github.com/ankurtaly/Integrated-Gradients

        It is meant to be used with a BERT classification or regression  model
        with the following characteristics.
        - The model is saved in the TF 1.0 SavedModel format
        - The model uses a single text segment as input. (Note that in general
          BERT models can be trained to take two separate text segments as
          input. However this class only works with models that take a single
          segment.)
        - The input text is tokenized and transformed using the standard BERT
          preprocessing. Specifically, the input is tokenized into wordpieces,
          with a special '[CLS]' token in the beginning and a
          special '[SEP]' token at the end, followed by a sequence of padding
          tokens ('[PAD]'). See https://arxiv.org/pdf/1609.08144.pdf for more
          details on WordPiece tokenization)

        If your model does not conform to any of the above characteristics then
        please  implement a specific  wrapper class for your model by
        inheriting from TFSavedModelWrapperIg.

        This class inherits from TFSavedModelWrapperIg and overrides the
        transform_input and project_attributions method from it.

        Args:
        :param saved_model_path: Path to the directory containing the BERT
            model in TF 1.0 SavedModel format.

        :param sig_def_key: Key for the specific SignatureDef to be used for
            executing the model.

        :param output_columns: A list containing the names of the output
            column(s) that corresponds to the output of the model. If the
            model is a binary classification model then the number of output
            columns is one, otherwise, the number of columns must match the
            shape of the output tensor corresponding to the output key
            specified.

        :param is_binary_classification [optional]: Boolean specifying if the
            model is a binary classification model. If True, the number of
            output columns is one. The default is False.

        :param output_key [optional]: Key for the specific output tensor (
            specified in the SignatureDef) whose predictions must be explained.
            The output tensor must specify a differentiable output of the
            model. Thus, output tensors that are generated as a result of
            discrete operations (e.g., argmax) are disallowed. The default is
            None, in which case the first output listed in the SignatureDef is
            used. The 'saved_model_cli' can be used to view the output tensor
            keys available in the signature_def.
            See: https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel

        :param batch_size [optional]: the batch size for input into the model.
            Depends on model and instance config.

        :param input_tensor_to_differentiable_layer_mapping [optional]:
            A dictionary mapping the input tensor where token ids are
            fed (typically named: 'input_ids') to the corresponding embedding
            tensor. The embedding tensor must be the addition of the token,
            position, and segment embedding tensor. There is a default value
            for this parameter based on the 'BERT-base, Uncased' model from
            https://github.com/google-research/bert
            Set this parameter ony if the names of these tensors are changed in
            your BERT model.

        :param max_seq_length [optional]: Maximum sequence length of input
            tokens. The default is 256.

        :param tokenizer_path [optional]: Path to tokenizer (pickle file) used
            to tokenize the input text. The default is a 'BERT-Base, Uncased'
            tokenizer.

        :param max_allowed_error [optional]: Float specifying a percentage
            value for the maximum allowed integral approximation error for IG
            computation. If None then IG will be  calculated for a
            pre-determined number of steps. Otherwise, the number of steps
            will be increased till the error is within the specified limit.

        :param word_level_attribution [optional]: Boolean specifying whether
            attributions should be returned at the word level. For models that
            require sentences to be tokenized at the sub-word level (e.g.,
            wordpiece or character level), setting  word_level_attribution to
            True would would cause a best effort aggregation of the sub-word
            level token  attributions at the word-level. If such an
            aggregation fails then the original attributions at the sub-word
            level would be returned. The  default  value of this  parameter
            is True.
        """
        super().__init__(
            saved_model_path,
            sig_def_key,
            is_binary_classification=is_binary_classification,
            output_key=output_key,
            batch_size=batch_size,
            output_columns=output_columns,
            input_tensor_to_differentiable_layer_mapping=
            input_tensor_to_differentiable_layer_mapping,
            max_allowed_error=max_allowed_error)

        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # When the word_level_attribution boolean is set to True,
        # the wordpiece attributions from the BERT model are aggregated at
        # the word-level. We use BERT's BasicTokenizer for word-level
        # tokenization, and preserve the casing in the original sentence.
        # This has the advantage that the aggregate attributions are tied to
        # word tokens as they appear in the sentence.
        self.word_tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
        self.max_seq_length = max_seq_length
        self.word_level_attribution = word_level_attribution

    def transform_input(self, input_df):
        """
        Transform the provided dataframe into one that complies with the input
        interface of the BERT model.

        Specifically, the BERT model takes four features as input: 'input_ids'
        'input_mask', 'segment_ids', 'label_ids'. This method derives the
        value of these features from the provided text segment in the input
        DataFrame.

        :param input_df: DataFrame with a single column named 'sentence' that
            contains the text whose prediction is being attributed.

        :returns transformed_input_df: DataFrame having four columns
            'input_ids', 'input_mask', 'segment_ids', 'label_ids' that specify
            the input to the BERT model.

        """
        transformed_input_df = input_df.copy(deep=True)
        transformed_input_df['input_ids'] = input_df['sentence'].apply(
            lambda x: ['[CLS]'] + self.tokenizer.tokenize(x)[:(
                self.max_seq_length - 2)] + ['[SEP]'])

        transformed_input_df['input_ids'] = transformed_input_df[
            'input_ids'].apply(
            lambda x: self.tokenizer.convert_tokens_to_ids(x))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        transformed_input_df['input_mask'] = transformed_input_df[
            'input_ids'].apply(lambda x: [1] * len(x))

        # Zero-pad up to the sequence length.
        transformed_input_df['input_ids'] = transformed_input_df[
            'input_ids'].apply(
            lambda x: x + [0] * (self.max_seq_length - len(x)))
        transformed_input_df['input_mask'] = transformed_input_df[
            'input_mask'].apply(
            lambda x: x + [0] * (self.max_seq_length - len(x)))
        transformed_input_df['segment_ids'] = transformed_input_df[
            'input_ids'].apply(lambda x: [0] * self.max_seq_length)
        transformed_input_df['label_ids'] = 0

        transformed_input_df = transformed_input_df[['input_ids',
                                                     'input_mask',
                                                     'segment_ids',
                                                     'label_ids']]
        return transformed_input_df

    def generate_baseline(self, input_df):
        """
        Generates a baseline for the provided input that is required for
        calculating Integrated Gradients.

        The returned baseline replaces each token in the provided input with a
        padding token ('[PAD]')

        :param input_df: DataFrame with a single column named 'sentence' that
            contains the text whose prediction is being attributed.

        :returns baseline_df: DataFrame having four columns
            'input_ids', 'input_mask', 'segment_ids', 'label_ids' that specify
            an input to the BERT classification models. These
            values specify a baseline input formed by replacing every token
            in the input sentence with a padding token.
        """
        assert(len(input_df) == 1)
        baseline_df = input_df.copy(deep=True)
        len_tokens = len(self.tokenizer.tokenize(input_df['sentence'][0]))
        baseline_df['input_ids'] = input_df['sentence'].apply(
            lambda x: ['[CLS]'] +
                      ['[PAD]'] * min(len_tokens, (self.max_seq_length - 2)) +
                      ['[SEP]'])

        # TODO(Aalok): We should be able to get rid of some of the following
        #  code by calling transform_input.
        baseline_df['input_ids'] = baseline_df['input_ids'].\
            apply(lambda x: self.tokenizer.convert_tokens_to_ids(x))

        baseline_df['input_mask'] = baseline_df['input_ids'].\
            apply(lambda x: [1] * len(x))

        # Zero-pad up to the sequence length.
        baseline_df['input_ids'] = baseline_df['input_ids'].\
            apply(lambda x: x + [0] * (self.max_seq_length - len(x)))

        baseline_df['input_mask'] = baseline_df['input_mask'].\
            apply(lambda x: x + [0] * (self.max_seq_length - len(x)))
        baseline_df['segment_ids'] = baseline_df['input_ids'].\
            apply(lambda x: [0] * self.max_seq_length)
        baseline_df['label_ids'] = 0

        baseline_df = baseline_df[['input_ids', 'input_mask', 'segment_ids',
                                   'label_ids']]
        return baseline_df

    def project_attributions(self, input_df, transformed_input_df,
                             attributions):
        """
        Maps the attributions to the token ids specified in the (transformed)
        input to the corresponding token texts.

        :param input_df: DataFrame with a single column named 'sentence' that
            contains the text whose prediction is being attributed.

        :param transformed_input_df: DataFrame specifying a BERT model input
            as returned by the transform_input function. It has exactly
            one row as currently only instance explanations are supported.

        :param attributions: dictionary with a single key 'input_ids' mapped
            to a list containing the attributions of the 'input_ids' (i.e.
            token ids)  specified in the transformed_input_df. The order is
            maintained.

        :returns a dictionary with a single key 'probabilities', mapped
            to a list of two lists, the first one containing the text tokens,
            and the second their corresponding attributions
        """
        def make_tokens_frontend_compatible(tokens):
            # The Fiddler front end currently requires each token to include
            # the space adjoining in the sentence.
            # TODO(Aalok, Ankur): This is ugly, we should get rid of it soon.
            return [t + ' ' for t in tokens]

        pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

        tokens = []
        token_attributions = []
        for i, t in enumerate(transformed_input_df['input_ids'][0]):
            if t in [pad_id, cls_id, sep_id]:
                continue
            tokens.append(self.tokenizer.convert_ids_to_tokens([int(t)])[0])
            token_attributions.append(
                attributions['input_ids'][0].astype('float')[i])

        if not self.word_level_attribution:
            # TODO(Aalok, Ankur): Front-end currently required spaces to be
            #  included as part of the tokens. This is a bit ugly and we should
            #  get rid of it.
            tokens = make_tokens_frontend_compatible(tokens)
            return {'probabilities': [tokens, token_attributions]}

        # Aggregate attributions at the word level.
        def tokenization_fn(sentence):
            return self.tokenizer.tokenize(sentence)

        sentence = input_df['sentence'][0]

        word_tokens = self.word_tokenizer.tokenize(sentence)
        word_covering = cover_tokens(
            coarse_grained_tokens=word_tokens,
            fine_grained_tokens=tokens,
            fine_grained_tokenization_fn=tokenization_fn)
        if word_covering is None:
            # A covering could not be constructed.
            logging.info(f'Failed to cover word-level tokens '
                         f'{word_tokens} with fine-grained tokens f'
                         f'{tokens}')
            tokens = make_tokens_frontend_compatible(tokens)
            return {'probabilities': [tokens, token_attributions]}

        # The covering comes with the guarantee that the concatenation of the
        # fine-grained tokens covering each word level token, recovers the
        # original list of tokens.
        offset = 0
        word_tokens = []
        word_attributions = []
        for word, covering_tokens in word_covering:
            word_tokens.append(word)
            word_attributions.append(np.sum(
                token_attributions[offset:offset+len(covering_tokens)]))
            offset += len(covering_tokens)
        word_tokens = make_tokens_frontend_compatible(word_tokens)
        return {'probabilities': [word_tokens,
                                  word_attributions]}
