import pandas as pd
import boto3
import logging
import json
import threading
import time
import yaml

_logger = logging.getLogger(__name__)


MAX_REQUEST_SIZE = 4 * 1024 * 1024  # Limit to 4MB. AWS API limit is 5MB.


class SageMakerRuntimeModel:
    """
    A Fiddler model package implementation that
    """
    def __init__(self, model_yaml_config):
        self.client = None
        self.client_refresh_time = 0  # Set when 'assume_role_arn' is set.
        self.lock = threading.Lock()
        with open(model_yaml_config) as f:
            config = yaml.safe_load(f)
        model = self._ensure_key(config, 'model')
        self.prediction_columns = [
            col['column-name'] for col in model.get('outputs', [])
        ]

        # Sagemaker related config
        endpoint_config = self._ensure_key(model, 'sagemaker_endpoint')
        self.endpoint_name = self._ensure_key(endpoint_config, 'endpoint_name')
        self.region_name = self._ensure_key(endpoint_config, 'region_name')
        self.aws_access_key_id = endpoint_config.get('aws_access_key_id', None)
        self.aws_secret_access_key = endpoint_config.get(
            'aws_secret_access_key', None)
        self.assume_role_arn = endpoint_config.get('assume_role_arn', None)
        self.refresh_client()

    @staticmethod
    def _ensure_key(config: dict, key: str):
        if key not in config:
            raise ValueError(f'"{key}" is not set in model configuration.')
        return config[key]

    def predict(self, input_df):

        num_rows = input_df.shape[0]
        _logger.info('Invoking endpoint with {} rows'.format(num_rows))
        results = []

        def row_chunks():
            # Split rows into chunks to keep size below MAX_REQUEST_SIZE.
            rows = []
            for idx in range(num_rows):
                total_size = 0
                df_slice = input_df.iloc[idx:idx+1]
                csv_row = df_slice.to_csv(header=False, index=False)
                if len(rows) > 0 and \
                        total_size + len(csv_row) >= MAX_REQUEST_SIZE:
                    yield rows
                    rows = []
                    total_size = 0
                total_size += len(csv_row)
                rows.append(csv_row)
            if len(rows) > 0:
                yield rows

        for csv_rows in row_chunks():
            self.refresh_client_if_required()
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Accept='application/json',
                Body=''.join(csv_rows)
            )

            for line in response.get('Body').iter_lines():
                # output format: {"predictions": [{"score": 2.043767929}]}
                for predictions in json.loads(line).get('predictions'):
                    results.append(predictions.get('score'))

        return pd.DataFrame(results, columns=self.prediction_columns)

    def refresh_client(self):
        _logger.info('Initializing sagemaker-runtime client')

        if self.assume_role_arn is not None:
            _logger.info(f'Fetching temporary credentials for '
                         f'role {self.assume_role_arn}')

            tmp_credentials = boto3.client(
                'sts',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            ).assume_role(
                RoleArn=self.assume_role_arn,
                RoleSessionName='Fiddler'
            )['Credentials']

            self.client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region_name,
                aws_access_key_id=tmp_credentials['AccessKeyId'],
                aws_secret_access_key=tmp_credentials['SecretAccessKey'],
                aws_session_token=tmp_credentials['SessionToken'])

            # tmp_credentials expire in 1 hour. Refresh 5 minutes before that.
            expires_at = tmp_credentials['Expiration']
            _logger.info(f'temporary credentials expire at {expires_at}')
            self.client_refresh_time = expires_at.timestamp() - 300
        else:
            self.client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key)
            self.client_refresh_time = float('inf')

    def refresh_client_if_required(self):
        if self.client_refresh_time < float('inf'):
            with self.lock:
                if time.time() >= self.client_refresh_time:
                    self.refresh_client()


if __name__ == '__main__':  # Quick Test
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s: %(message)s')
    import os
    df = pd.read_csv(os.environ['INPUT'], header=None)
    model_yaml = os.environ['MODEL_YAML']
    model = SageMakerRuntimeModel(model_yaml)
    print(model.predict(df))
