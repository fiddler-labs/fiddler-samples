# TODO: Add License
from collections import namedtuple
import contextlib
import copy
import json
import logging
import os.path
from pathlib import Path
import pickle
import shutil
import tempfile
import textwrap
from typing import Any, Dict, List, Iterable, Optional, Sequence, Union
from urllib.parse import urlparse
import yaml

import pandas as pd
import requests
import configparser

from .core_objects import (
    AttributionExplanation,
    DatasetInfo,
    ModelInfo,
    ModelInputType,
    ModelTask,
    MulticlassAttributionExplanation,
)
from .utils import df_from_json_rows, _try_series_retype

LOG = logging.getLogger()

SUCCESS_STATUS = 'SUCCESS'
FAILURE_STATUS = 'FAILURE'
FIDDLER_ARGS_KEY = '__fiddler_args__'
STREAMING_HEADER_KEY = 'X-Fiddler-Results-Format'
AUTH_HEADER_KEY = 'Authorization'
ROUTING_HEADER_KEY = 'x-fdlr-fwd'
ADMIN_SERVICE_PORT = 4100

# A PredictionEventBundle represents a batch of inferences and their input
# features. All of these share schema, latency, and success status. A bundle
# can consist just one event as well.
PredictionEventBundle = namedtuple(
    'PredictionEventBundle',
    [
        'prediction_status',  # typeof: int # 0 for success, failure otherwise
        'prediction_latency',  # typeof: float # Latency in seconds.
        'input_feature_bundle',  # list of feature vectors.
        'prediction_bundle',  # list of prediction vectors.
        # TODO: Support sending schema as well.
    ],
)


_protocol_version = 1


class FiddlerApi:
    """Broker of all connections to the Fiddler API.
    Conventions:
        - Exceptions are raised for FAILURE reponses from the backend.
        - Methods beginning with `list` fetch lists of ids (e.g. all model ids
            for a project) and do not alter any data or state on the backend.
        - Methods beginning with `get` return a more complex object but also
            do not alter data or state on the backend.
        - Methods beginning with `run` invoke model-related execution and
            return the result of that computation. These do not alter state,
            but they can put a heavy load on the computational resources of
            the Fiddler engine, so they should be paralellized with care.
        - Methods beginning with `delete` permanently, irrevocably, and
            globally destroy objects in the backend. Think "rm -rf"
        - Methods beginning with `upload` convert and transmit supported local
            objects to Fiddler-friendly formats loaded into the Fiddler engine.
            Attempting to upload an object with an identifier that is already
            in use will result in an exception being raised, rather than the
            existing object being overwritten. To update an object in the
            Fiddler engine, please call both the `delete` and `upload` methods
            pertaining to the object in question.

    :param url: The base URL of the API to connect to. Usually either
        https://api.fiddler.ai (cloud) or http://localhost:4100 (onebox)
    :param org_id: The name of your organization in the Fiddler platform
    :param auth_token: Token used to authenticate. Your token can be
        created, found, and changed at <FIDDLER ULR>/settings/credentials.
    :param verbose: if True, api calls will be logged verbosely
    """

    def __init__(self, url=None, org_id=None, auth_token=None, verbose=False):
        if Path('fiddler.ini').is_file():
            config = configparser.ConfigParser()
            config.read('fiddler.ini')
            info = config['FIDDLER']
            if not url:
                url = info['url']
            if not org_id:
                org_id = info['org_id']
            if not auth_token:
                auth_token = info['auth_token']

        if url[-1] == '/':
            raise ValueError('url should not end in "/"')

        # use session to preserve session data
        self.session = requests.Session()
        self.url = url
        self.org_id = org_id
        self.auth_header = {AUTH_HEADER_KEY: f'Bearer {auth_token}'}
        self.streaming_header = {STREAMING_HEADER_KEY: 'application/jsonlines'}
        self.verbose = verbose
        LOG.info(f'url ={url}')
        try:
            _ = self.list_projects()
        except requests.exceptions.ConnectionError:
            LOG.warning(
                f'CONNECTION CHECK FAILED: Unable to connect with to '
                f'{self.url}. Are you sure you have the right URL?'
            )
        except RuntimeError as error:
            LOG.warning(
                f'API CHECK FAILED: Able to connect to {self.url}, '
                f'but request failed with message "{str(error)}"'
            )

    @staticmethod
    def _get_routing_header(path_base: str) -> Dict[str, str]:
        """Returns the proper header so that a request is routed to the correct
        service."""
        executor_service_bases = (
            'dataset_predictions',
            'execute',
            'executor',
            'explain',
            'explain_by_row_id',
            'feature_importance',
            'generate',
            'new_project',
            'v1/point_explanation',
            'v1/model_explanation',
            'v1/execution',
        )
        if path_base in executor_service_bases:
            return {ROUTING_HEADER_KEY: 'executor_service'}
        else:
            return {ROUTING_HEADER_KEY: 'data_service'}

    def _call(
        self,
        path: List[str],
        json_payload: Any = None,
        files: Optional[List[Path]] = None,
        is_get_request: bool = False,
        stream: bool = False,
    ):
        """Issues a request to the API and returns the result,
        logigng and handling errors appropriately.


        Raises a RuntimeError if the response is a failure or cannot be parsed.
        Does not handle any ConnectionError exceptions thrown by the `requests`
        library.
        """

        # set up a context manager to open files
        with contextlib.ExitStack() as context_stack:
            endpoint = '/'.join([self.url] + path)
            request_type = 'GET' if is_get_request else 'POST'
            if json_payload:
                request_excerpt = textwrap.indent(
                    json.dumps(json_payload, indent=2)[:2048], '  '
                )
            else:
                request_excerpt = None
            if self.verbose:
                LOG.info(
                    f'running api call as {request_type} request\n'
                    f'to {endpoint}\n'
                    f'with headers {self.auth_header}\n'
                    f'with payload {request_excerpt}'
                )
            if is_get_request:
                req = requests.Request('GET', endpoint)
            else:
                # if uploading files, we use a multipart/form-data request and
                # dump the json_payload to be the special "fiddler args"
                # as a json object in the form
                if files is not None:
                    # open all the files into the context manager stack
                    opened_files = {
                        fpath.name: context_stack.enter_context(fpath.open('rb'))
                        for fpath in files
                    }
                    '''
                    NOTE: there are a lot LOT of ways to do this wrong with
                    `requests`

                    Take a look here (and at the thread linked) for more info:
                    https://stackoverflow.com/a/35946962

                    And here: https://stackoverflow.com/a/12385661
                    '''
                    form_data = {
                        **{
                            FIDDLER_ARGS_KEY: (
                                None,  # filename
                                json.dumps(json_payload),  # data
                                'application/json',  # content_type
                            )
                        },
                        **{
                            fpath.name: (
                                fpath.name,  # filename
                                opened_files[fpath.name],  # data
                                'application/octet-stream',  # content_type
                            )
                            for fpath in files
                        },
                    }
                    req = requests.Request('POST', endpoint, files=form_data)
                else:
                    req = requests.Request('POST', endpoint, json=json_payload)

            # add necessary headers
            # using prepare_request from session to keep session data
            req = self.session.prepare_request(req)
            added_headers = dict()
            added_headers.update(self.auth_header)
            added_headers.update(self._get_routing_header(path[0]))
            if stream:
                added_headers.update(self.streaming_header)
            req.headers = {**added_headers, **req.headers}

            # log the raw request
            raw_request_info = (
                f'Request:\n'
                f'  url: {req.url}\n'
                f'  method: {req.method}\n'
                f'  headers: {req.headers}'
            )
            LOG.debug(raw_request_info)
            # if self.verbose:
            #     LOG.info(raw_request_info)

            # send the request using session to carry auth info from login
            res = self.session.send(req, stream=stream)

            if self.verbose:
                LOG.info(f'response: {res.text}')

        # catch auth failure
        if res.status_code == 401:
            error_msg = (
                f'API call to {endpoint} failed with status 401: '
                f'Authorization Required. '
                f'Do you have the right org_id and auth_token?'
            )
            LOG.debug(error_msg)
            raise RuntimeError(error_msg)
        # catch any failure
        elif res.status_code != 200:
            error_msg = (
                f'API call to {endpoint} failed with status {res.status_code}:'
                f' The full response message was {res.text}'
            )
            LOG.debug(error_msg)
            raise RuntimeError(error_msg)

        if stream:
            return self._process_streaming_call_result(res, endpoint, raw_request_info)
        return self._process_non_streaming_call_result(res, endpoint, raw_request_info)

    @staticmethod
    def _raise_on_status_error(
        response: requests.Response, endpoint: str, raw_request_info: str
    ):
        """Raises exception on HTTP errors similar to
         `response.raise_for_status()`. """
        # catch non-auth failures
        try:
            response.raise_for_status()
        except Exception:
            response_payload = response.json()
            try:
                failure_message = response_payload.get('message', 'Unknown')
                failure_stacktrace = response_payload.get('stacktrace')
                error_msg = (
                    f'API call failed.\n'
                    f'Error message: {failure_message}\n'
                    f'Endpoint: {endpoint}'
                )
                if failure_stacktrace:
                    error_msg += f'\nStacktrace: {failure_stacktrace}'

            except KeyError:
                error_msg = (
                    f'API call to {endpoint} failed.\n'
                    f'Request response: {response.text}'
                )
            LOG.debug(f'{error_msg}\n{raw_request_info}')
            raise RuntimeError(error_msg)

    def _process_non_streaming_call_result(
        self, response: requests.Response, endpoint: str, raw_request_info: str
    ):

        FiddlerApi._raise_on_status_error(response, endpoint, raw_request_info)

        # catch non-JSON response (this is rare, the backend should generally
        # return JSON in all cases)
        try:
            response_payload = response.json()
        except json.JSONDecodeError:
            print(response.status_code)
            error_msg = (
                f'API call to {endpoint} failed.\n' f'Request response: {response.text}'
            )
            LOG.debug(f'{error_msg}\n{raw_request_info}')
            raise RuntimeError(error_msg)

        assert response_payload['status'] == SUCCESS_STATUS
        result = response_payload.get('result')

        # log the API call on success (excerpt response on success)
        response_excerpt = textwrap.indent(
            json.dumps(response_payload, indent=2)[:2048], '  '
        )
        log_msg = (
            f'API call to {endpoint} succeeded.\n'
            f'Request response: {response_excerpt}\n'
            f'{raw_request_info}\n'
        )
        if self.verbose:
            LOG.info(log_msg)
        return result

    @staticmethod
    def _process_streaming_call_result(
        response: requests.Response, endpoint: str, raw_request_info: str
    ):
        """Processes response in jsonlines format. `json_streaming_endpoint`
           returns jsonlines with one json object per line when
           'X-Fiddler-Response-Format' header is set to 'jsonlines'.
           :returns: a generator for results."""

        FiddlerApi._raise_on_status_error(response, endpoint, raw_request_info)

        got_eos = False  # got proper end_of_stream.

        if response.headers.get('Content-Type') != 'application/x-ndjson':
            RuntimeError('Streaming response Content-Type is not "x-ndjson"')

        # Read one line at a time. `chunk_size` None ensures that a line
        # is returned as soon as it is read, rather waiting for any minimum
        # size (default is 512 bytes).
        for line in response.iter_lines(chunk_size=None):
            if line:
                row_json = json.loads(line)
                if 'result' in row_json:
                    yield row_json['result']
                elif row_json.get('status') == SUCCESS_STATUS:
                    got_eos = True
                    break
        if not got_eos:
            raise RuntimeError(
                'Truncated response for streaming request. '
                'Failed to receive successful status.'
            )

    def list_datasets(self) -> List[str]:
        """List the ids of all datasets in the organization.

        :returns: List of strings containing the ids of each dataset.
        """
        try:
            path = ['v1', 'dataset', 'get', self.org_id]
            payload = {}
            res = self._call(path, json_payload=payload)
        except Exception:
            path = ['list_datasets', self.org_id]
            res = self._call(path, is_get_request=True)

        return res

    def list_projects(self) -> List[str]:
        """List the ids of all projects in the organization.

        :returns: List of strings containing the ids of each project.
        """
        try:
            path = ['v1', 'project', 'get', self.org_id]
            payload = {}
            res = self._call(path, json_payload=payload)
        except Exception:
            path = ['list_projects', self.org_id]
            res = self._call(path, is_get_request=True)

        return res

    def list_models(self, project_id: str, cached=True) -> List[str]:
        """List the names of all models in a project.

        :param project_id: The unique identifier of the project on the Fiddler
            engine.
        :param cached: Receive a fresh, uncached list. Used for testing.

        :returns: List of strings containing the ids of each model in the
            specified project.
        """
        try:
            path = ['v1', 'model', 'get', self.org_id]
            payload = {'project_id': project_id, 'cached': cached}
            res = self._call(path, json_payload=payload)
        except Exception:
            path = ['list_models', self.org_id, project_id]
            res = self._call(path, is_get_request=True)

        return res

    def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """Get DatasetInfo for a dataset.

        :param dataset_id: The unique identifier of the dataset on the Fiddler
            engine.

        :returns: A fiddler.DatasetInfo object describing the dataset.
        """
        try:
            path = ['v1', 'dataset', 'get', self.org_id]
            payload = {'dataset_id': dataset_id}
            res = self._call(path, json_payload=payload)
        except Exception:
            path = ['dataset_schema', self.org_id, dataset_id]
            res = self._call(path, is_get_request=True)

        info = DatasetInfo.from_dict(res)
        info.dataset_id = dataset_id
        return info

    def get_model_info(self, project_id: str, model_id: str) -> ModelInfo:
        """Get ModelInfo for a model in a certain project.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.

        :returns: A fiddler.ModelInfo object describing the model.
        """
        try:
            path = ['v1', 'model', 'get', self.org_id]
            payload = {'project_id': project_id, 'model_id': model_id}
            res = self._call(path, json_payload=payload)
        except Exception:
            path = ['model_info', self.org_id, project_id, model_id]
            res = self._call(path, is_get_request=True)
        return ModelInfo.from_dict(res)

    def _query_dataset(
        self,
        dataset_id: str,
        fields: List[str],
        max_rows: int,
        split: Optional[str] = None,
        stream=True,
        sampling=False,
        sampling_seed=0.0,
    ):
        payload = dict(
            streaming=True,
            dataset_id=dataset_id,
            fields=fields,
            limit=max_rows,
            sampling=sampling,
        )

        if sampling:
            payload['sampling_seed'] = sampling_seed
        if split is not None:
            payload['source'] = f'{split}.csv'
        # try:
        #     path = ['v1', 'dataset', 'get', self.org_id]
        #     res = self._call(path, json_payload=payload, stream=stream)
        # except Exception:
        payload.pop('streaming')
        payload.pop('dataset_id')

        path = ['dataset_query', self.org_id, dataset_id]
        res = self._call(path, json_payload=payload, stream=stream)

        return res

    def get_dataset(
        self,
        dataset_id: str,
        max_rows: int = 1_000,
        splits: Optional[List[str]] = None,
        sampling=False,
        dataset_info: Optional[DatasetInfo] = None,
        include_fiddler_id=False,
        stream=True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetches data from a dataset on Fiddler.

        :param dataset_id: The unique identifier of the dataset on the Fiddler
            engine.
        :param max_rows: Up to this many rows will be fetched from eash split
            of the dataset.
        :param splits: If specified, data will only be fetched for these
            splits. Otherwise, all splits will be fetched.
        :param sampling: If True, data will be sampled up to max_rows. If
            False, rows will be returned in order up to max_rows. The seed
            will be fixed for sampling.∂
        :param dataset_info: If provided, the API will skip looking up the
            DatasetInfo (a necessary precursor to requesting data).
        :param include_fiddler_id: Return the Fiddler engine internal id
            for each row. Useful only for debugging.
        :param stream: Streaming is generally faster, but you can disable
            this if you are having errors. Does not affect the results
            returned.

        :returns: A dictionary of str -> DataFrame that maps the name of
            dataset splits to the data in those splits. If len(splits) == 1,
            returns just that split as a dataframe, rather than a dataframe.
        """
        if dataset_info is None:
            dataset_info = self.get_dataset_info(dataset_id)
        else:
            dataset_info = copy.deepcopy(dataset_info)

        def get_df_from_split(split):
            dataset_rows = self._query_dataset(
                dataset_id,
                fields=dataset_info.get_column_names(),
                max_rows=max_rows,
                split=split,
                sampling=sampling,
                stream=stream,
            )
            return df_from_json_rows(
                dataset_rows, dataset_info, include_fiddler_id=include_fiddler_id
            )

        if splits is None:
            use_splits = [
                os.path.splitext(filename)[0] for filename in dataset_info.files
            ]
        else:
            use_splits = splits
        res = {split: get_df_from_split(split) for split in use_splits}
        if splits is not None and len(splits) == 1:
            # unwrap single-slice results
            res = next(iter(res.values()))
        return res

    def get_slice(
        self,
        sql_query: str,
        project: Optional[str],
        columns_override: Optional[List[str]] = None,
        stream=True,
    ) -> pd.DataFrame:
        """Fetches data from Fiddler via a *slice query* (SQL query).

        :param sql_query: A special SQL query that begins with the keyword
            "SLICE"
        :param project: A project is required when the the slice query is
            for a model. The caller might not know if the query is a
            model slice. Is it safe to provide it whenever it is available.
        :param columns_override: A list of columns to return even if they are
            not specified in the slice.
        :param stream: Streaming is generally faster, but you can disable
            this if you are having errors. Does not affect the results
            returned.

        :returns: A table containing the sliced data (as a Pandas DataFrame)
        """
        payload = dict(streaming=True, sql=sql_query, project=project)
        if columns_override is not None:
            payload['slice_columns_override'] = columns_override

        try:
            path = ['v1', 'dataset', 'get', self.org_id]
            res = self._call(path, json_payload=payload, stream=stream)
        except Exception:
            payload.pop('streaming')

            path = ['slice_query', self.org_id]
            res = self._call(path, json_payload=payload, stream=stream)

        if stream:
            try:
                slice_info = next(res)
            except StopIteration:
                raise RuntimeError('Empty results for slice!')
        else:
            slice_info = res.pop(0)
        if not slice_info['is_slice']:
            raise RuntimeError(
                'Query does not return a valid slice. ' 'Query: ' + sql_query
            )
        column_names = slice_info['columns']
        dtype_strings = slice_info['dtypes']
        df = pd.DataFrame(res, columns=column_names)
        for column_name, dtype in zip(column_names, dtype_strings):
            df[column_name] = _try_series_retype(df[column_name], dtype)
        return df

    def delete_dataset(self, dataset_id: str):
        """Permanently delete a dataset.

        :param dataset_id: The unique identifier of the dataset on the Fiddler
            engine.

        :returns: Server response for deletion action.
        """
        try:
            path = ['v1', 'dataset', 'delete', self.org_id]
            payload = {'dataset_id': dataset_id}
            result = self._call(path, json_payload=payload)
        except Exception:
            path = ['dataset_delete', self.org_id, dataset_id]
            result = self._call(path)

        return result

    def delete_model(
        self, project_id: str, model_id: str, delete_prod=False, delete_pred=True
    ):
        """Permanently delete a model.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.
        :param delete_prod: Boolean value to delete the production table.
            By default this table is not dropped.
        :param delete_pred: Boolean value to delete the prediction table.
            By default this table is dropped.

        :returns: Server response for deletion action.
        """
        try:
            path = ['v1', 'model', 'delete', self.org_id]
            payload = {
                'project_id': project_id,
                'model_id': model_id,
                'delete_prod': delete_prod,
                'delete_pred': delete_pred,
            }
            result = self._call(path, json_payload=payload)
        except Exception:
            path = ['delete_model', self.org_id, project_id, model_id]
            result = self._call(path)

        return result

    def delete_model_artifacts(self, project_id: str, model_id: str):
        """Permanently delete a model artifacts.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.

        :returns: Server response for deletion action.
        """
        # delete from executor service cache
        path = ['delete_model_artifacts', self.org_id, project_id, model_id]
        result = self._call(path)

        return result

    def delete_project(self, project_id: str):
        """Permanently delete a project.

        :param project_id: The unique identifier of the project on the Fiddler
            engine.

        :returns: Server response for deletion action.
        """
        try:
            path = ['v1', 'project', 'delete', self.org_id]
            payload = {'project_id': project_id}
            result = self._call(path, json_payload=payload)
        except Exception:
            path = ['delete_project', self.org_id, project_id]
            result = self._call(path)

        return result

    def _upload_dataset_csvs(
        self,
        dataset_id: str,
        csv_file_paths: List[Path],
        split_test: Optional[bool] = None,
        dataset_info: Optional[DatasetInfo] = None,
    ):
        """Uploads a CSV file to the Fiddler platform."""
        payload = dict(dataset_name=dataset_id)
        if split_test is not None:
            payload['split_test'] = split_test
        if dataset_info is not None:
            payload['dataset_info'] = dict(dataset=dataset_info.to_dict())

        try:
            path = ['v1', 'dataset', 'upload', self.org_id]
            result = self._call(path, json_payload=payload, files=csv_file_paths)
        except Exception:
            path = ['dataset_upload', self.org_id]
            result = self._call(path, json_payload=payload, files=csv_file_paths)
        return result

    def _db_import(self, dataset_id):
        try:
            path = ['v1', 'dataset', 'import', self.org_id]
            payload = dict(dataset_id=dataset_id)
            res = self._call(path, json_payload=payload)
        except Exception:
            path = ['import_dataset', self.org_id, dataset_id]
            res = self._call(path, is_get_request=True)

        return res

    def upload_dataset(
        self,
        dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        dataset_id: str,
        info: Optional[DatasetInfo] = None,
        split_test=False,
    ):
        """Uploads a dataset to the Fiddler engine.

        :param dataset: A DataFrame or dictionary mapping name -> DataFrame
            containing tabular data to be uploaded to the Fiddler engine.
        :param dataset_id: The unique identifier of the dataset on the Fiddler
            engine. Must be a short string without whitespace.
        :param info: A DatasetInfo object specifying all the details of the
            dataset. If not provided, a DatasetInfo will be inferred from the
            dataset and a warning raised.
        :param split_test: If you would like Fiddler to automatically split a
            single-dataframe dataset into a training set and a test set, set
            `split_test=True`. This option has no effect for multi-dataframe
            datasets.

        :returns: The server response for the upload.
        """
        assert (
            ' ' not in dataset_id
        ), 'The dataset identifier should not contain whitespace'

        # get a dictionary of str -> pd.DataFrame for all data to upload
        if not isinstance(dataset, dict):
            dataset = dict(data=dataset)

        # infer a dataset_info
        inferred_info = DatasetInfo.from_dataframe(
            dataset.values(), display_name=dataset_id
        )

        if info:
            # Since we started populating stats recently, some older yamls
            # dont have it. Or the user might just supply us the basic
            # schema without stats.
            # If the user provided the schema/yaml file, lets make sure
            # that we add stats if they are not already there.
            info = DatasetInfo.update_stats_for_existing_schema(dataset, info)
            updated_infos = []
            for item in dataset.values():
                update_info = DatasetInfo.check_and_update_column_info(info, item)
                updated_infos.append(update_info)
            info = DatasetInfo.as_combination(
                updated_infos, display_name=info.display_name
            )
        # validate `info` if passed
        if info is not None:
            inferred_columns = inferred_info.get_column_names()
            passed_columns = info.get_column_names()
            if inferred_columns != passed_columns:
                raise RuntimeError(
                    f'Provided data schema has columns {passed_columns}, '
                    f'which does not match the data schema {inferred_columns}'
                )
        # use inferred info with a warning if not `info` is passed
        else:
            LOG.warning(
                f'Heads up! We are inferring the details of your dataset from '
                f'the dataframe(s) provided. Please take a second to check '
                f'our work.'
                f'\n\nIf the following DatasetInfo is an incorrect '
                f'representation of your data, you can construct a '
                f'DatasetInfo with the DatasetInfo.from_dataframe() method '
                f'and modify that object to reflect the correct details of '
                f'your dataset.'
                f'\n\nAfter constructing a corrected DatasetInfo, please '
                f're-upload your dataset with that DatasetInfo object '
                f'explicitly passed via the `info` parameter of '
                f'FiddlerApi.upload_dataset().'
                f'\n\nYou may need to delete the initially uploaded version'
                f'via FiddlerApi.delete_dataset(\'{dataset_id}\').'
                f'\n\nInferred DatasetInfo to check:'
                f'\n{textwrap.indent(repr(inferred_info), "  ")}'
            )
            info = inferred_info

        # determine whether or not the index of this dataset is a meaningful
        # column that should be written into CSV files
        include_index = next(iter(dataset.values())).index.name is not None

        # dump CSVs to named temp file
        with tempfile.TemporaryDirectory() as tmp:
            csv_paths = list()
            for name, df in dataset.items():
                filename = f'{name}.csv'
                path = Path(tmp) / filename
                csv_paths.append(path)
                LOG.info(f'[{name}] dataset upload: writing csv to {path}')
                df.to_csv(path, index=include_index)

            # add files to the DatasetInfo on the fly
            new_schema = copy.deepcopy(info)
            new_schema.files = [path.name for path in csv_paths]

            # upload the CSV
            LOG.info(f'[{dataset_id}] dataset upload: uploading csv')
            self._upload_dataset_csvs(
                dataset_id, csv_paths, split_test=split_test, dataset_info=new_schema
            )

        # trigger dataset upload
        LOG.info(f'[{dataset_id}] dataset upload: ' f'triggering import of dataset')
        res = self._db_import(dataset_id)
        return res

    def upload_dataset_from_dir(self, dataset_id: str, dataset_dir: Path):
        logging.info(f'uploading dataset from dir: {dataset_dir}')
        if not dataset_dir.is_dir():
            raise ValueError(f'{dataset_dir} is not a directory')
        dataset_yaml = dataset_dir / f'{dataset_id}.yaml'
        if not dataset_yaml.is_file():
            raise ValueError(f'YAML file not found: {dataset_yaml}')
        with dataset_yaml.open() as f:
            dataset_info = DatasetInfo.from_dict(yaml.safe_load(f))
            files = dataset_dir.glob('*.csv')
            csv_files = [x for x in files if x.is_file()]
            logging.info(f'Found CSV file {csv_files}')
            self._upload_dataset_csvs(dataset_id, csv_files, False, dataset_info)
            result = self._db_import(dataset_id)
            LOG.info(f'Dataset uploaded {result}')

    def run_model(
        self, project_id: str, model_id: str, df: pd.DataFrame, log_events=False
    ) -> pd.DataFrame:
        """Executes a model in the Fiddler engine on a DataFrame.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.
        :param df: A dataframe contining model inputs as rows.
        :param log_events: Variable determining if the the predictions
            generated should be logged as production traffic

        :returns: A pandas DataFrame containing the outputs of the model.
        """
        data_array = [y.iloc[0, :].to_dict() for x, y in df.groupby(level=0)]
        payload = dict(
            project_id=project_id,
            model_id=model_id,
            data=data_array,
            logging=log_events,
        )

        try:
            path = ['v1/execution', 'run', self.org_id]
            res = self._call(path, json_payload=payload)
        except Exception:
            payload.pop('project_id')
            payload.pop('model_id')

            path = ['execute', self.org_id, project_id, model_id]
            res = self._call(path, json_payload=payload)
        return pd.DataFrame(res)

    def run_explanation(
        self,
        project_id: str,
        model_id: str,
        df: pd.DataFrame,
        explanations: Union[str, Iterable[str]] = 'shap',
        dataset_id: Optional[str] = None,
    ) -> Union[
        AttributionExplanation,
        MulticlassAttributionExplanation,
        List[AttributionExplanation],
        List[MulticlassAttributionExplanation],
    ]:
        """Executes a model in the Fiddler engine on a DataFrame.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.
        :param df: A dataframe containing model inputs as rows. Only the first
            row will be explained.
        :param explanations: A single string or list of strings specifying
            which explanation algorithms to run.
        :param dataset_id: The unique identifier of the dataset in the
            Fiddler engine. Required for most tabular explanations, but
            optional for most text explanations.

        :returns: A single AttributionExplanation if `explanations` was a
            single string, or a list of AttributionExplanation objects if
            a list of explanations was requested.
        """
        """Explains a model's prediction on a single instance"""
        # wrap single explanation name in a list for the API
        if isinstance(explanations, str):
            explanations = (explanations,)

        data_array = [y.iloc[0, :].to_dict() for x, y in df.groupby(level=0)]

        payload = dict(
            project_id=project_id,
            model_id=model_id,
            data=data_array[0],
            explanations=[dict(explanation=ex) for ex in explanations],
        )
        if dataset_id is not None:
            payload['dataset'] = dataset_id

        try:
            path = ['v1/point_explanation', 'run', self.org_id]
            res = self._call(path, json_payload=payload)

        except Exception:
            payload.pop('project_id')
            payload.pop('model_id')

            path = ['explain', self.org_id, project_id, model_id]
            res = self._call(path, json_payload=payload)

        explanations_list = res['explanations']
        # TODO: enable more robust check for multiclass explanations
        is_multiclass = 'explanation' not in explanations_list[0]
        deserialize_fn = (
            MulticlassAttributionExplanation.from_dict
            if is_multiclass
            else AttributionExplanation.from_dict
        )
        ex_objs = [
            deserialize_fn(explanation_dict) for explanation_dict in explanations_list
        ]
        if len(ex_objs) == 1:
            return ex_objs[0]
        else:
            return ex_objs

    def run_feature_importance(
        self,
        project_id: str,
        model_id: str,
        dataset_id: str,
        dataset_splits: Optional[List[str]] = None,
        slice_query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get global feature importance for a model over a dataset.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.
        :param dataset_id: The unique identifier of the dataset in the
            Fiddler engine.
        :param dataset_splits: If specified, importance will only be computed
            over these splits. Otherwise, all splits will be used. Only a
            single split is currently supported.
        :param kwargs: Additional parameters to be passed to the importance
            algorithm. For example, `n_iterations`, `n_references`, and
            `ci_confidence_level`.
        :return: A named tuple with the explanation results.
        """
        if dataset_splits is not None and len(dataset_splits) > 1:
            raise NotImplementedError(
                'Unfortunately, only a single split is '
                'currently supported for feature '
                'importances.'
            )
        source = None if dataset_splits is None else dataset_splits[0]

        payload = dict(
            subject='feature_importance',
            project_id=project_id,
            model_id=model_id,
            dataset_id=dataset_id,
            source=source,
            slice_query=slice_query,
        )
        payload.update(kwargs)

        try:
            path = ['v1/model_explanation', 'run', self.org_id]
            res = self._call(path, json_payload=payload)
        except Exception:
            payload.pop('subject')
            payload.pop('project_id')
            payload.pop('model_id')
            payload['dataset'] = payload.pop('dataset_id')

            path = ['feature_importance', self.org_id, project_id, model_id]
            res = self._call(path, json_payload=payload)
        # wrap results into named tuple
        res = namedtuple('FeatureImportanceResults', res)(**res)
        return res

    def create_project(self, project_id: str):
        """Create a new project.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine. Must be a short string without whitespace.

        :returns: Server response for creation action.
        """
        res = None
        try:
            path = ['v1', 'project', 'run', self.org_id]
            payload = {'project_id': project_id}
            res = self._call(path, json_payload=payload)
        except RuntimeError:
            path = ['new_project', self.org_id, project_id]
            res = self._call(path)
        except Exception as e:
            if 'already exists' in str(e):
                print('Project already exists, no change.')
            else:
                raise e

        if res:
            return res

    def create_model(
        self,
        project_id: str,
        dataset_id: str,
        target: str,
        features: Optional[List[str]] = None,
        train_splits: Optional[List[str]] = None,
        model_id: str = 'fiddler_generated_model',
    ):
        """Trigger auto-modeling on a dataset already uploaded to Fiddler.

        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.

        :param dataset_id: The unique identifier of the dataset in the
            Fiddler engine.
        :param target: The column name of the target to be modeled.
        :param features: If specified, a list of column names to use as
            features. If not specified, all non-target columns will be used
            as features.
        :param train_splits: A list of splits to train on. If not specified,
            all splits will be used as training data. Currently only a single
            split can be specified if `train_splits` is not None.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine.

        :returns: Server response for creation action.
        """
        if train_splits is not None:
            if len(train_splits) == 0:
                raise ValueError(
                    '`train_splits` cannot be an empty list. Pass `None` if '
                    'you want to train on the entire dataset.'
                )
            if len(train_splits) > 1:
                raise NotImplementedError(
                    'Sorry, currently only single-split training is '
                    'supported. Please only pass a maximum of one element to '
                    '`train_splits`.'
                )
        source = f'{train_splits[0]}.csv'
        dataset_column_names = self.get_dataset_info(dataset_id).get_column_names()

        # raise exception if misspelled target
        if target not in dataset_column_names:
            raise ValueError(
                f'Target "{target}" not found in the columns of '
                f'dataset {dataset_id} ({dataset_column_names})'
            )

        # raise if target in features or features not in columns
        if features is not None:
            if target in features:
                raise ValueError(f'Target "{target}" cannot also be in ' f'features.')
            features_not_in_dataset = set(features) - set(dataset_column_names)
            if len(features_not_in_dataset) > 0:
                raise ValueError(
                    f'All features should be in the dataset, but '
                    f'the following features were not found in '
                    f'the dataset: {features_not_in_dataset}'
                )

        # use all non-target columns from dataset if no features are specified
        if features is None:
            features = list(dataset_column_names)
            features.remove(target)

        payload = {
            'project_id': project_id,
            'dataset_id': dataset_id,
            'model_id': model_id,
            'source': source,
            'inputs': features,
            'targets': [target],
        }

        try:
            path = ['v1', 'training', 'run', self.org_id]
            result = self._call(path, json_payload=payload)
        except Exception:
            payload.pop('project_id')
            payload['dataset'] = payload.pop('dataset_id')
            payload['model_name'] = payload.pop('model_id')

            path = ['generate', self.org_id, project_id]
            result = self._call(path, json_payload=payload)
        return result

    def upload_model_sklearn(
        self,
        model,
        info: ModelInfo,
        project_id: str,
        model_id: str,
        associated_dataset_ids: Optional[List[str]] = None,
    ):
        """Uploads a subclass of sklearn.base.BaseEstimator to the Fiddler
        engine.

        :param model: An instance of a sklearn.base.BaseEstimator object to be
            uploaded. NOTE: custom subclasses are not supported.
        :param info: A ModelInfo object describing the details of the model.
        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine. Must be a short string without
            whitespace.
        :param associated_dataset_ids: The unique identifiers of datasets in
            the Fiddler engine to associate with the model.

        :returns: Server response for upload action.
        """
        assert ' ' not in model_id, 'The model identifier should not contain whitespace'

        model_info = FiddlerApi._add_dataset_ids_to_model_info(
            info, associated_dataset_ids
        )

        # add framework info in ModelInfo
        try:
            sklearn_version_number = model.__getstate__()['_sklearn_version']
            info.framework = f'scikit-learn {sklearn_version_number}'
        except KeyError:
            pass

        sklearn_upload_warning = (
            'You are uploading a scikit-learn model using the Fiddler API.'
            '\nIf this model uses any custom (non-sklearn) code, it will not '
            'run properly on the Fiddler Engine.'
            '\nThe Fiddler engine may not be able to detect this in advance.'
        )
        LOG.warning(sklearn_upload_warning)

        payload = dict(
            project=project_id,
            model=model_id,
            model_schema=dict(model=model_info.to_dict()),
            framework=info.framework,
            model_type='sklearn',
        )
        LOG.info(f'[{model_id}] model upload: uploading pickle')
        with tempfile.TemporaryDirectory() as tmp:
            pickle_path = Path(tmp) / 'model.pkl'
            with pickle_path.open('wb') as pickle_file:
                pickle_file.write(pickle.dumps(model))

                try:
                    endpoint_path = ['v1', 'model', 'upload', self.org_id]
                    result = self._call(
                        endpoint_path, json_payload=payload, files=[pickle_path]
                    )
                except Exception:
                    endpoint_path = ['model_upload', self.org_id]
                    result = self._call(
                        endpoint_path, json_payload=payload, files=[pickle_path]
                    )
                return result

    def upload_model_custom(
        self,
        artifact_path: Path,
        info: ModelInfo,
        project_id: str,
        model_id: str,
        associated_dataset_ids: Optional[List[str]] = None,
    ):
        """Uploads a custom model object to the Fiddler engine along with
            custom glue-code for running the model.

        :param artifact_path: A path to a directory containing all of the
            model artifacts needed to run the model. This includes a
            `package.py` file with the glue code needed to run the model.
        :param info: A ModelInfo object describing the details of the model.
        :param project_id: The unique identifier of the model's project on the
            Fiddler engine.
        :param model_id: The unique identifier of the model in the specified
            project on the Fiddler engine. Must be a short string without
            whitespace.
        :param associated_dataset_ids: The unique identifiers of datasets in
            the Fiddler engine to associate with the model.

        :returns: Server response for upload action.
        """
        assert ' ' not in model_id, 'The model identifier should not contain whitespace'

        if not artifact_path.is_dir():
            raise ValueError(f'The {artifact_path} must be a directory.')

        model_info = FiddlerApi._add_dataset_ids_to_model_info(
            info, associated_dataset_ids
        )

        # upload the model
        payload = dict(
            project=project_id,
            model=model_id,
            model_schema=dict(model=model_info.to_dict()),
            framework=info.framework,
            upload_as_archive=True,
            model_type='custom',
        )
        with tempfile.TemporaryDirectory() as tmp:
            tarfile_path = Path(tmp) / 'files'
            shutil.make_archive(
                base_name=str(Path(tmp) / 'files'),
                format='tar',
                root_dir=str(artifact_path),
                base_dir='.',
            )
            LOG.info(
                f'[{model_id}] model upload: uploading custom model from'
                f' artifacts in {str(artifact_path)} tarred to '
                f'{str(tarfile_path)}'
            )

            try:
                endpoint_path = ['v1', 'model', 'upload', self.org_id]
                result = self._call(
                    endpoint_path, json_payload=payload, files=[Path(tmp) / 'files.tar']
                )
            except Exception:
                endpoint_path = ['model_upload', self.org_id]
                result = self._call(
                    endpoint_path, json_payload=payload, files=[Path(tmp) / 'files.tar']
                )
            return result

    def upload_model_package(
        self, artifact_path: Path, project_id: str, model_id: str,
    ):
        if not artifact_path.is_dir():
            raise ValueError(f'not a valid model dir: {artifact_path}')
        yaml_file = artifact_path / 'model.yaml'
        if not yaml_file.is_file():
            raise ValueError(f'Model yaml not found {yaml_file}')
        with yaml_file.open() as f:
            model_info = ModelInfo.from_dict(yaml.safe_load(f))
            self.upload_model_custom(artifact_path, model_info, project_id, model_id)

    def upload_mlflow_model(
        self,
        mlflow_uri: str,
        project_id: str,
        model_id: str,
        dataset_id: Optional[str] = None,
        target_col_name: Optional[str] = None,
        feature_col_names: Optional[Sequence[str]] = None,
        input_type: Optional[ModelInputType] = ModelInputType.TABULAR,
        model_task: Optional[ModelTask] = None,
        model_info: Optional[str] = None,
    ):
        """
        `mlflow_uri` must be to a local file, e.g. file:///path/to/model

        If `model_info` is given, `dataset_id`, `target_col_name`, `
            feature_col_names`, `input_type`, and `model_task` are ignored.
            Otherwise, they are required and will be used to infer a
            ModelInfo object.
        """
        assert ' ' not in model_id, 'The model identifier should not contain whitespace'

        with tempfile.TemporaryDirectory() as tmp:
            # tar the MLFlow model directory into a temp dir
            tmp_dir = Path(tmp)
            mlflow_model_dir = Path(urlparse(mlflow_uri).path)
            LOG.info(f'Tarring MLFlow model directory {mlflow_model_dir}')
            model_tarfile = shutil.make_archive(
                base_name=str(tmp_dir / 'files'),
                format='gztar',
                root_dir=str(mlflow_model_dir.parent),
                base_dir=str(mlflow_model_dir.name),
            )
            model_tarfile = Path(model_tarfile)

            # construct a ModelInfo if none exists
            if model_info is None:
                missing_required = (
                    dataset_id is None
                    or target_col_name is None
                    or feature_col_names is None
                )
                if missing_required:
                    raise ValueError(
                        'You must either provide a `model_info`'
                        ' or provide all of the following: `dataset_id`, '
                        '`target_col_name`, `feature_col_names`.'
                    )
                dataset_info = self.get_dataset_info(dataset_id)
                model_info = ModelInfo.from_dataset_info(
                    dataset_info,
                    target_col_name,
                    feature_col_names,
                    input_type=input_type,
                    model_task=model_task,
                )
                # add dataset id to model info misc params
                model_info = FiddlerApi._add_dataset_ids_to_model_info(
                    model_info, [dataset_id]
                )

            # upload the model
            endpoint_path = ['mlflow_upload', self.org_id]
            payload = dict(
                project=project_id,
                model=model_id,
                model_schema=dict(model=model_info.to_dict()),
            )
            LOG.info(
                f'[{model_id}] model upload: uploading mlflow model from'
                f' {mlflow_model_dir} as {model_tarfile}'
            )
            return self._call(
                endpoint_path, json_payload=payload, files=[model_tarfile]
            )

    def setup_monitoring(
        self, project_id, name, baseline_df, dataset_info, target, features
    ):
        try:
            self.get_dataset_info(name)
            raise ValueError(f'name already in used: {name}')
        except Exception:
            LOG.info('dataset not found, creating it')

        self.upload_dataset(baseline_df, name, split_test=False, info=dataset_info)
        train_splits = ['data']
        self.create_model(
            project_id,
            name,
            target=target,
            features=features,
            train_splits=train_splits,
            model_id=name,
        )
        project_url = self.url.replace('host.docker.internal', 'localhost', 1)
        return (
            f'Monitoring successfully setup on Fiddler. \n '
            f'Visit {project_url}/projects/{project_id} to monitor'
        )

    @staticmethod
    def _add_dataset_ids_to_model_info(model_info, associated_dataset_ids):
        model_info = copy.deepcopy(model_info)
        # add associated dataset ids to ModelInfo
        if associated_dataset_ids is not None:
            for dataset_id in associated_dataset_ids:
                assert (
                    ' ' not in dataset_id
                ), 'Dataset identifiers should not contain whitespace'
            model_info.misc['datasets'] = associated_dataset_ids
        return model_info

    def trigger_dataset_import(self, dataset_id: str):
        """Makes the Fiddler service (re-)import the specified dataset."""
        return self._db_import(dataset_id)

    def trigger_model_predictions(
        self, project_id: str, model_id: str, dataset_id: str
    ):
        """Makes the Fiddler service compute and cache model predictions on a
        dataset."""
        payload = dict(project_id=project_id, model_id=model_id, dataset_id=dataset_id)

        try:
            path = ['v1', 'execution', 'run', self.org_id]
            result = self._call(path, json_payload=payload)
        except Exception:
            payload.pop('project_id')
            payload['dataset'] = payload.pop('dataset_id')

            result = self._call(
                ['dataset_predictions', self.org_id, project_id], payload
            )

        return result

    def publish_event(
        self,
        project_id: str,
        model_id: str,
        event: dict,
        event_id: Optional[str] = None,
        update_event: Optional[bool] = None,
        event_time_stamp: Optional[int] = None,
    ):
        """
       Publishes an event to Fiddler Service.
       :param project_id: the project to which the model whose events are
                          being published belongs
       :param model_id: the model whose events are being published
       :param dict event: Dictionary of event details, such as features
                          and predictions.
       :param event_id: unique str event id for the event
       :param update_event: if the event is an update to a previously
                            published row
       :param event_time_stamp: the UTC timestamp of the event in
                                milliseconds

       """
        if update_event:
            event['__event_type'] = 'update_event'
            event['__updated_at'] = event_time_stamp
            if event_id is None:
                raise ValueError('An update event needs an event_id')
        else:
            event['__event_type'] = 'execution_event'
            event['__occurred_at'] = event_time_stamp

        if event_id is not None:
            event['__event_id'] = event_id

        path = ['external_event', self.org_id, project_id, model_id]
        result = self._call(path, event)

        return result
