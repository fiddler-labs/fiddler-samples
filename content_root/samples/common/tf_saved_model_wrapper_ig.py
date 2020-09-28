from tf_saved_model_wrapper import TFSavedModelWrapper
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

        # TODO: Add an example in next PRs

        Args:
        :param saved_model_path: Path to the directory containing the TF
            model in SavedModel format.
            See: https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel

        :param sig_def_key: Key for the specific SignatureDef to be used for
            executing the model.
            See: https://www.tensorflow.org/tfx/serving/signature_defs#signaturedef_structure

        :param output_columns: List containing the names of the output
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
        """
        Generates a DataFrame specifying a baseline that is required for
        calculating Integrated Gradients.

        The Baseline is a certain 'informationless' input relative to which
        attributions must be computed. For instance, in a text
        classification model, the baseline could be the empty text.

        The baseline could be the same for all inputs or could be specific
        to the input at hand. An example of the latter would be a baseline
        containing as many padding tokens as the number of tokens in the
        input text.


        The choice of baseline is important as explanations are contextual to a
        baseline. For more information please refer to the following document:
        https://github.com/ankurtaly/Integrated-Gradients/blob/master/howto.md
        """
        raise NotImplementedError('Please implement generate_baseline in '
                                  'package.py')

    def project_attributions(self, input_df, transformed_input_df,
                             attributions):
        """
        Maps the attributions for the provided transformed_input to
        the original untransformed input.

        This method returns a dictionary mapping features of the untransformed
        input to the untransformed feature value, and (projected) attributions
        computed for that feature.

        This method guarantees that for each feature the projected attributions
        have the same shape as the (returned) untransformed feature value. The
        specific projection being applied is left as an implementation detail.
        Below we provided some guidance on the projections that should be
        applied for three different transformations

        Identity transformation
        This is the simplest case. Since the transformation is identity, the
        projection would also be the identity function.

        One-hot transformation for categorical features
        Here the original feature is categorical, and the transformed feature
        is a one-hot encoding. In this case, the returned untransformed feature
        value is the specific input category, and the projected attribution is
        the sum of the attribution across all fields of the one-hot encoding.

        Token ID transformation for text features
        Here the original feature is a sentence, and transformed feature is a
        vector of token ids (w.r.t.a certain vocabulary). Here the
        untransformed feature value would be a vector of tokens corresponding
        to the token ids, and the projected attribution vector would be the
        same as the one provided to this method. In some cases, token ids
        corresponding to dummy token such a padding tokens, start tokens, end
        tokens, etc. may be ignored during the projection. In that case, the
        attributions values  corresponding to these tokens must be dropped from
        the projected attributions vector.

        :param input_df: Pandas DataFrame specifying the input whose prediction
            is being attributed. Its columns must correspond to the dataset
            yaml associated with the project. Specifically, the columns must
            correspond to the feature names mentioned in the yaml.

        :param transformed_input_df: Pandas DataFrame returned by the
            transform_input method extended in package.py. It has exactly
            one row as currently only instance explanations are supported.

        :param attributions: dictionary mapping each column of the
            transformed_input to the corresponding attributions tensor. The
            attribution tensor must have the same shape as corresponding
            column in transformed_input.

        Returns:
        - projected_inputs: dictionary with keys being the features of the
            original untransformed input. The features are specified in the
            model.yaml. The keys are mapped to a pair containing the original
            untransformed input and the projected attribution.
        """
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
