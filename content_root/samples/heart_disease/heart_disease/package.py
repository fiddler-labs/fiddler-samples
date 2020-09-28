import pathlib
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class MyModel:
    def __init__(self, max_allowed_error=None,
                 output_columns=['predicted_target']):
        self.max_allowed_error = max_allowed_error

        model_dir = pathlib.Path(__file__).parent

        self.sess = tf.Session()
        with self.sess.as_default():
            self.model = load_model(pathlib.Path(model_dir) /
                                    'heart_disease_num_features.h5')
        self.ig_enabled = True
        self.is_input_differentiable = True
        self.batch_size = 256
        self.output_columns = output_columns
        self.input_tensors = self.model.input
        self.output_tensor = self.model.output
        self.gradient_tensors = \
            {'predicted_target':
                 {self.input_tensors:
                      tf.gradients(self.output_tensor, self.input_tensors)}}
        self.input_tensor_to_differentiable_layer_mapping = {
            self.input_tensors: self.input_tensors}
        self.differentiable_tensors = {self.input_tensors: self.input_tensors}

    def get_feed_dict(self, input_df):
        """
        Returns the input dictionary to be fed to the TensorFlow graph given
        input_df which is a pandas DataFrame. The input_df DataFrame is
        obtained after applying transform_input on the raw input. The
        transform_input function is extended in package.py.
        """

        feed = {self.input_tensors: input_df.values}
        return feed

    def transform_input(self, input_df):
        return input_df

    def generate_baseline(self, input_df):
        return input_df*0

    def predict(self, input_df):
        transformed_input_df = self.transform_input(input_df)

        with self.sess.as_default():

            predictions = self.model.predict(transformed_input_df)

        return pd.DataFrame(data=predictions, columns=self.output_columns)

    def project_attributions(self, input_df, transformed_input_df,
                             attributions):
        return {col: attributions[self.input_tensors][0][i].tolist()
                for i, col in enumerate(input_df.columns)}


def get_model():
    model = MyModel(max_allowed_error=1)
    return model
