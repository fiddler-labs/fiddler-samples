import logging
import os
import pandas as pd
import pickle as pkl
import xgboost as xgb


class SageMakerXGBoostPredictor:
    """An XGBoost predictor for a sagemaker xgboost saved model.
       This loads the predictor once and runs for each call to predict.
    """

    def __init__(self, model_path, output_column=None):
        """
        :param model_path: The directory where the model is saved.
        :param output_column: list of column name(s) for the output.
        """
        self.model_path = model_path
        self.output_column = output_column

        self.model = pkl.load(open(self.model_path, 'rb'))

    def predict(self, input_df):
        input_df.columns = self.model.feature_names
        dtest = xgb.DMatrix(input_df)
        pred = self.model.predict(dtest)
        return pd.DataFrame(pred, columns=self.output_column)


# Manual testing.
def main():
    input_df = pd.read_csv('lending_test_package.csv', index_col=0)
    input_df = input_df.drop(columns=['loan_status'])
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost-model')
    model = SageMakerXGBoostPredictor(model_path,
                                      output_column=['loan_status'])
    result = model.predict(input_df)
    print(result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s: %(message)s')
    main()
