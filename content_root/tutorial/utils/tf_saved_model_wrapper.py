
import numpy as np
import pandas as pd
import tensorflow as tf
import logging


class TFSavedModelWrapper:
    def __init__(self, saved_model_path, sig_def_key, output_columns,
                 is_binary_classification=False, output_key=None,
                 batch_size=8):
        """
        Wrapper to load and run a TF model from a saved_model path.
        Models must extend this class in their package.py, and override the
        transform_input method.

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
        """

        self.saved_model_path = saved_model_path
        self.sig_def_key = sig_def_key
        self.output_key = output_key
        self.output_columns = output_columns
        self.input_tensors = None
        self.output_tensor = None
        self.sess = None
        self.saved_model = None
        self.is_binary_classification = is_binary_classification
        self.batch_size = batch_size

    def load_model(self):
        """
        Loads the model and creates a session from the saved_model_path
        provided at initialization.
        """
        # load the model
        self.sess = tf.Session()
        self.saved_model = tf.saved_model.loader.load(
            sess=self.sess, tags=['serve'],
            export_dir=str(self.saved_model_path))

        # Extract input and output tensors from the signature.
        sig = self.saved_model.signature_def[self.sig_def_key]
        self.input_tensors = sig.inputs

        if self.output_key is None:
            self.output_key = list(sig.outputs)[0]

        self.output_tensor = self.get_tensor(sig.outputs[self.output_key].name)
        if self.is_binary_classification:
            if len(self.output_columns) != 1:
                raise ValueError(f'Number of output columns should be one '
                                 f'for a binary classification model, '
                                 f'but length is {len(self.output_columns)} ')
            # output_tensor should either be of shape <batch, > or <batch, 2>
            output_tensor_shape = self.output_tensor.shape.as_list()
            logging.info(f'Output tensor shape is {output_tensor_shape}')
            if len(output_tensor_shape) == 2:
                if output_tensor_shape[1] == 2:
                    self.output_tensor = self.output_tensor[:, 1]

    def transform_input(self, input_df):
        raise NotImplementedError('Please implement transform_input in package.py')

    def predict(self, input_df):
        """
        Returns predictions for the provided inputs.

        Args:
        :param input_df: DataFrame corresponding to the dataset yaml
            associated with the project. Specifically, the columns in the
            DataFrame must correspond to the feature names mentioned in the
            yaml.

        Returns:
        - predictions_df: Pandas DataFrame with predictions for the provided
            inputs. The columns of the DataFrame are the provided set of output
            columns.
        """

        transformed_input_df = self.transform_input(input_df)
        predictions = []
        for ind in range(0, len(transformed_input_df), self.batch_size):
            df_chunk = transformed_input_df.iloc[ind: ind + self.batch_size]
            feed = self.get_feed_dict(df_chunk)

            with self.sess.as_default():
                predictions += self.sess.run(self.output_tensor, feed).tolist()
        return pd.DataFrame(predictions, columns=self.output_columns)

    def get_tensor(self, name):
        names = [t.name for t in self.sess.graph.as_graph_def().node]
        try:
            return self.sess.graph.get_tensor_by_name(name)
        except:
            raise ValueError(f'name: {name} not in graph: {names}')

    def get_feed_dict(self, input_df):
        """
        Returns the input dictionary to be fed to the TensorFlow graph given
        input_df which is a pandas DataFrame. The input_df DataFrame is
        obtained after applying transform_input on the raw input. The
        transform_input function is extended in package.py.
        """

        feed = {}
        for key, tensor_info in self.input_tensors.items():
            if key not in input_df.columns:
                raise RuntimeError(f'Transformed input does not have a '
                                   f'column corresponding to the input tensor '
                                   f'key: {key} specified in the SignatureDef. Input col: {input_df.columns}')
            feed_inp = input_df[key].tolist()
            feed_inp_shape = np.array(feed_inp).shape
            expected_shape = self.get_shape(tensor_info.tensor_shape)
            if not self.match_shape(feed_inp_shape, expected_shape):
                raise RuntimeError(f'Shape mismatch for input tensor {key}.'
                                   f'Got: {feed_inp_shape}, Want '
                                   f'{expected_shape}')
            feed[tensor_info.name] = feed_inp
        return feed

    @staticmethod
    def get_shape(tensor_shape):
        """
        Returns shape of tensor having tensor shape in the format returned by
        the SignatureDef
        """
        return [d.size for d in tensor_shape.dim]

    @staticmethod
    def get_shape_tensor(tensor_shape):
        """
        Returns shape of tensor having tensor shape in the format of the
        tf.TensorShape class
        """
        return [d.value if d.value is not None else -1 for d in
                tensor_shape.dims]

    @staticmethod
    def match_shape(got, want):
        if len(got) != len(want):
            return False
        for i, v in enumerate(got):
            if want[i] != -1 and want[i] != v and v != -1:
                return False
        return True
