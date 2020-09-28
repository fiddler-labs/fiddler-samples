import pandas as pd
import requests
from io import StringIO


class IrisPMMLModel:

    def load_model(self):
        pass

    def transform_input(self, input_df):
        return input_df

    def predict(self, input_df):
        # convert input to csv
        csv_input = input_df.to_csv()

        # call openscoring server,
        # replace this URL with your openscoring endpoint
        OPEN_SCORING_URL = 'http://host.docker.internal:8080/' \
                           'openscoring/model/DecisionTreeIris/csv'
        headers = {'Content-type': 'text/plain; charset=UTF-8'}
        res = requests.post(
            OPEN_SCORING_URL,
            data=csv_input,
            headers=headers
        )

        # Drop extra columns from openscoring response
        df = pd.read_csv(StringIO(res.text), sep=',')
        df.drop(columns=['Species', 'Node_Id'], inplace=True)
        return df


def get_model_class():
    return IrisPMMLModel()
