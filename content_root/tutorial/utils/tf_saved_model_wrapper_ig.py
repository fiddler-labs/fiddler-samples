
from .tf_saved_model_wrapper import TFSavedModelWrapper
import tensorflow as tf
import logging


class TFSavedModelWrapperIg(TFSavedModelWrapper):
    def __init__(self, saved_model_path, sig_def_key, output_columns,
                 is_binary_classification=False,
                 output_key=None,
                 batch_size=8,
                 input_tensor_to_differentiable_layer_mapping={},
                 max_allowed_error=None):
        """
        Wrapper to support Integrated Gradients (IG) computation for a TF
        model loaded from a saved_model path.

        See: https://github.com/ankurtaly/Integrated-Gradients

        Models must extend this class in their  package.py, and override the
        transform_input and the project_attributions methods.

        Args:
        :param input_tensor_to_differentiable_layer_mapping [optional]:
            Dictionary that maps input tensors to the first differentiable
            layer/tensor in the graph they are attached to. For instance,
            in a text model, an input tensor containing token ids
            may not be differentiable but may feed into an embedding tensor.
            Such an input tensor must be mapped to the corresponding the
            embedding tensor in this dictionary.

            All input tensors must be mentioned in the dictionary. An input
            tensor that is directly differentiable may be mapped to itself.

            For each differentiable tensor, the first dimension must be the
            batch dimension. If <k1, …, kn> is the shape of the input then the
            differentiable tensor must either have the same shape or the shape
            <k1, …, kn, d>.

            The default is None, in which case all input tensors are assumed
            to be differentiable.

        :param max_allowed_error: Float specifying a percentage value
            for the maximum allowed integral approximation error for IG
            computation. If None then IG will be  calculated for a
            pre-determined number of steps. Otherwise, the number of steps
            will be increased till the error is within the specified limit.
        """

        super().__init__(saved_model_path, sig_def_key,
                         output_columns=output_columns,
                         is_binary_classification=is_binary_classification,
                         output_key=output_key,
                         batch_size=batch_size)

        self.input_tensor_to_differentiable_layer_mapping = \
            input_tensor_to_differentiable_layer_mapping

        # mapping from each input tensor to its differentiable version
        self.differentiable_tensors = {}

        # mapping each output column to a dictionary of gradients tensors.
        self.gradient_tensors = {}
        self.steps = 10  # no of steps for ig calculation
        self.ig_enabled = True  #
        self.max_allowed_error = max_allowed_error

    def load_model(self):
        """Extends load model defined in the TFSavedModelWrapper class"""
        super().load_model()

        for key, tensor_info in self.input_tensors.items():
            if key in self.input_tensor_to_differentiable_layer_mapping.keys():
                differentiable_tensor = \
                    self.get_tensor(
                        self.input_tensor_to_differentiable_layer_mapping[key])
                # shape check
                diff_tensor_shape = \
                    self.get_shape_tensor(differentiable_tensor.shape)
                input_tensor_shape = self.get_shape(tensor_info.tensor_shape)

                logging.info(f'For key {key} differentiable tensor shape is '
                             f'{diff_tensor_shape} input tensor shape is '
                             f'{input_tensor_shape}')
                if self._validate_differentiable_tensor_shape(
                        diff_tensor_shape, input_tensor_shape):
                    self.differentiable_tensors[key] = \
                        differentiable_tensor
                else:
                    raise ValueError(f'Shape of differentiable tensor '
                                     f'{diff_tensor_shape} doesnt follow rule '
                                     f'"If <k1, …, kn> is the shape of the '
                                     f'input then the differentiable tensor '
                                     f'must either have the same shape or the '
                                     f'shape <k1, …, kn, d>". Shape of input '
                                     f'tensor is {input_tensor_shape}')

        if self.is_binary_classification:
            self.gradient_tensors[self.output_columns[0]] = {}
            for key, tensor in self.differentiable_tensors.items():
                self.gradient_tensors[self.output_columns[0]][key] = \
                    tf.gradients(self.output_tensor, tensor)
        else:
            for index, column in enumerate(self.output_columns):
                self.gradient_tensors[column] = {}
                for key, tensor in self.differentiable_tensors.items():
                    self.gradient_tensors[column][key] = \
                        tf.gradients(self.output_tensor[:, index], tensor)

    def generate_baseline(self, input_df):
        raise NotImplementedError('Please implement generate_baseline in '
                                  'package.py')

    def project_attributions(self, input_df, transformed_input_df,
                             attributions):
        raise NotImplementedError('Please implement project_attributions in '
                                  'package.py')

    def _validate_differentiable_tensor_shape(self,
                                              differentiable_tensor_shape,
                                              input_tensor_shape):

        diff_len = len(differentiable_tensor_shape)
        input_len = len(input_tensor_shape)
        if diff_len == input_len:
            return self.match_shape(differentiable_tensor_shape,
                                    input_tensor_shape)
        elif diff_len - input_len == 1:
            return self.match_shape(differentiable_tensor_shape[:-1],
                                    input_tensor_shape)

        return False
