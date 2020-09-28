import logging
import os
import pandas as pd
import re

from tensorflow.python.client import session
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader
from tensorflow.python.tools import saved_model_utils

# Default signature def key and tag for TensorFlow serving.
# These could be made configurable for more custom saved models.
DEFAULT_SIGNATURE_DEF_KEY = 'predict'
DEFAULT_TAG = 'serve'

# Output: Only single 'predictions' output is expected, which is an ndarray.
# We might need to support more custom forms of outputs, in addition to default for serving.
DEFAULT_OUTPUT_KEY = 'predictions'


class TensorFlowPredictor:
    """A TensorFlow predictor for saved model.
       This loads the graph and model weights once and runs the
       session for each call to predict. The output is typically the generic output
       as returned in TensorFlow Serving (called "predictions")
    """
    # The implementation is similar to how TensorFlow's 'saved_model_cli' runs
    # saved modes. For more information see
    # https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel

    def __init__(self, model_dir, output_columns=None):
        """
        :param model_dir: The directory where the model is saved.
        :param output_columns: List of column names for the output.
                The saved models typically return single output called "predictions"
                containing all the predictions as a 2-d numpy array where number of columns
                is expected number of outputs for each input tuple. 'output_columns' provides
                names for each of the columns. If it is None, default names are assigned.
        """
        self.model_dir = model_dir
        self.output_columns = output_columns

        self.meta_graph_def = saved_model_utils.get_meta_graph_def(
            self.model_dir,
            tag_set=DEFAULT_TAG)
        signature_def = self.meta_graph_def.signature_def[
            DEFAULT_SIGNATURE_DEF_KEY]
        self.input_tensors = signature_def.inputs

        output_tensors = signature_def.outputs

        # Output is expected to be single "predictions" ndarray. Enforce that.
        if output_tensors.keys() != {DEFAULT_OUTPUT_KEY}:
            raise RuntimeError(
                'Expected single output named "{}", but found [{}]'.format(
                    DEFAULT_OUTPUT_KEY, ','.join(output_tensors.keys()))
            )

        self.output_tensor = output_tensors[DEFAULT_OUTPUT_KEY]
        output_shape = tuple(
            d.size for d in self.output_tensor.tensor_shape.dim)

        # Ensure that the output_tensor shape matches (-1, len(output_columns))
        if self.output_columns:
            expected_shape = (-1, len(self.output_columns))
            if expected_shape != output_shape:
                raise RuntimeError(
                    'Shape of prediction does not match with output columns. '
                    'Expected shape is {}, but found {}.'.format(
                        expected_shape,
                        output_shape))
        else:
            self.output_columns = [f'prediction_{i}' for i in
                                   range(output_shape[1])]

        self.sess = session.Session(None, graph=ops_lib.Graph())
        loader.load(self.sess, [DEFAULT_TAG], self.model_dir)

    def predict(self, input_df):
        # Fix column names: replace white space with '_'.
        # We could also check if the names match input_tensors here.
        # These will be checked during inference anyway.
        input_df = input_df.rename(
            mapper=lambda col: re.sub('\\s', '_', col),
            axis='columns',
            copy=False)
        input_feed_dict = {   # column tensor name => column values in input
            self.input_tensors[col].name: input_df[col]
            for col in input_df.columns
        }
        results = self.sess.run(self.output_tensor.name, feed_dict=input_feed_dict)
        return pd.DataFrame(results, columns=self.output_columns)

    def unload_model(self):
        # TODO: This is currently not called by the executor. It should.
        if self.sess:
            sess = self.sess
            self.sess = None
            sess.close()


# Manual testing.
def main():
    input_df = pd.read_csv('/tmp/x-sample.csv')
    model_dir = os.path.join(os.path.dirname(__file__), 'dnn_model')
    model = TensorFlowPredictor(model_dir, ['quality'])
    results = model.predict(input_df)
    print(results)
    model.unload_model()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s: %(message)s')
    main()
