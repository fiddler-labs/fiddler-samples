from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd
import tensorflow as tf


class TFSavedModel:
    def __init__(
            self,
            saved_model_path: Union[str, Path],
            output_column_names: Sequence[str],
            is_binary_classification: bool = False,
            batch_size: int = 8,
            input_transformation:
            Callable[[pd.DataFrame], Sequence[np.ndarray]] =
            lambda input_df: [input_df.values]):
        """Load and run a TF model saved in a saved_model format.

        NOTE: The model is loaded using the default serving signature def.

        Args:
        :param saved_model_path: Path to the directory containing the TF
            model in SavedModel format.
            See: https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel  # noqa E501

        :param output_column_names: List containing the names of the output
            column(s) that corresponds to the output of the model. If the
            model is a binary classification model then the number of output
            columns must be one, even if the output tensor has two columns.
            Otherwise, the number of columns must match the
            shape of the output tensor.

         :param is_binary_classification [optional]: Boolean specifying if the
            model is a binary classification model. If True, the number of
            output columns is one. The default is False.

        :param batch_size [optional]: the batch size for input into the model.
            Depends on model and instance config.

        :param input_transformation [optional]: This function transforms
            a DataFrame of inputs into a sequence of arrays/tensors that can be
            fed into the input tensors of the model.

            NOTE: The default input_transformation assumes the model has a
                single input that accepts the array df.values. A custom
                input_transformation must be provided if a more complicated
                transformation is needed between DataFrame and model inputs.
            NOTE: If the model has multiple inputs, the order of the elements
                of the sequence returned by input_transformation should match
                the order of the inputs in the model's signature def.
        """
        self.saved_model_path = saved_model_path
        self.output_column_names = output_column_names
        self.batch_size = batch_size
        self.is_binary_classification = is_binary_classification
        self.input_transformation = input_transformation

        # load the model
        self.sess = tf.Session()
        saved_model = tf.saved_model.loader.load(sess=self.sess,
                                                 tags=['serve'],
                                                 export_dir=str(
                                                     self.saved_model_path))

        # load the signature def
        sig_def = saved_model.signature_def[tf.saved_model.signature_constants.
                                            DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # find model input names from the signature def
        self.input_tensors = [
            self.sess.graph.get_tensor_by_name(sig_def.inputs[inp].name)
            for inp in sig_def.inputs
        ]

        # identify the output tensor from the signature def
        #  using only the positive class output in binary classification
        if len(sig_def.outputs) != 1:
            raise RuntimeError(
                f'Only single-output models are supported, but this model has '
                f'{len(sig_def.outputs)}.')
        self.output_tensor = self.sess.graph.get_tensor_by_name(
            sig_def.outputs[next(iter(sig_def.outputs))].name)
        if is_binary_classification and self.output_tensor.shape[1] == 2:
            self.output_tensor = self.output_tensor[:, 1]

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Returns predictions for the provided inputs.

        Args:
        :param input_df: DataFrame of input features.

        :return: DataFrame with predictions for the provided inputs.
            The columns of the DataFrame correspond in name and order
            to the model's output_column_names.
        """
        input_values_sequence = self.input_transformation(input_df)
        n_row = input_df.shape[0]
        n_col = len(self.output_column_names)
        pred_array = np.empty((n_row, n_col))
        batch_slice_gen = (slice(ind, ind + self.batch_size)
                           for ind in range(0, n_row, self.batch_size))
        with self.sess.as_default():
            for batch_slice in batch_slice_gen:
                feed_dict = {
                    model_input.name: input_values_tensor[batch_slice]
                    for model_input, input_values_tensor in zip(
                        self.input_tensors, input_values_sequence)
                }
                pred_array[batch_slice] = self.sess.run(
                    self.output_tensor, feed_dict)
        return pd.DataFrame(pred_array, columns=self.output_column_names)


class KerasModel:
    def __init__(
        self,
        saved_model_path: Union[str, Path],
        output_column_names: Sequence[str],
        is_binary_classification: bool = False,
        input_transformation: Callable[
            [pd.DataFrame],
            Sequence[np.ndarray]] = lambda input_df: [input_df.values]):
        """Load and run a TF model saved in a saved_model format.
        NOTE: The model is loaded using the default serving signature def.
        Args:
        :param saved_model_file_path: Path to the .h5 model file.
        :param output_column_names: List containing the names of the output
            column(s) that corresponds to the output of the model. If the
            model is a binary classification model then the number of output
            columns must be one, even if the output tensor has two columns.
            Otherwise, the number of columns must match the
            shape of the output tensor.
        :param input_transformation [optional]: This function transforms
            a DataFrame of inputs into a sequence of arrays/tensors that can be
            fed into the input tensors of the model.
            NOTE: The default input_transformation assumes the model has a
                single input that accepts the array df.values. A custom
                input_transformation must be provided if a more complicated
                transformation is needed between DataFrame and model inputs.
            NOTE: If the model has multiple inputs, the order of the elements
                of the sequence returned by input_transformation should match
                the order of the inputs in the model's signature def.
        """
        self.saved_model_path = saved_model_path
        self.output_column_names = output_column_names
        self.input_transformation = input_transformation
        self.is_binary_classification = is_binary_classification
        self.sess = tf.Session()
        with self.sess.as_default():
            self.model = tf.keras.models.load_model(str(self.saved_model_path))
        if self.is_binary_classification and len(self.output_column_names) > 1:
            raise ValueError(
                f'For binary classification, please only provide one output '
                f'column name, not {len(self.output_column_names)}')

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Returns predictions for the provided inputs.
        Args:
        :param input_df: DataFrame of input features.
        :return: DataFrame with predictions for the provided inputs.
            The columns of the DataFrame correspond in name and order
            to the model's output_column_names.
        """
        with self.sess.as_default():
            pred_array = self.model.predict(
                self.input_transformation(input_df))
        if self.is_binary_classification and pred_array.shape[1] == 2:
            pred_array = pred_array[:, 1]
        return pd.DataFrame(pred_array, columns=self.output_column_names)
