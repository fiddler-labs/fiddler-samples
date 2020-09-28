import logging
import os
import pandas as pd
import pickle as pkl


class RFPredictor:
    """An Randomforest predictor for bank_churn data.
       This loads the predictor once and runs for each call to predict.
    """

    def __init__(self, model_path, output_column=None):
        """
        :param model_path: The directory where the model is saved.
        :param output_column: list of column name(s) for the output.
        """
        self.model_path = model_path
        self.output_column = output_column

        file_path = os.path.join(self.model_path, 'model.pkl')
        with open(file_path, 'rb') as file:
            self.model = pkl.load(file)

    def predict(self, input_df):
        return pd.DataFrame(
            self.model.predict_proba(input_df)[:, 0], columns=self.output_column
        )


# Manual testing.
def main():
    input_df = pd.read_csv(os.environ['INPUT'], index_col=0)
    input_df = input_df.drop(columns=['Churned'])
    model_path = os.path.dirname(__file__)
    model = RFPredictor(model_path, output_column=['Churned'])
    model.load_model()
    result = model.predict(input_df)
    print(result)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-7s: %(message)s'
    )
    main()
