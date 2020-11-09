import copy
import logging
from typing import List

import pandas as pd
import numpy as np

from .core_objects import (
    Column,
    DatasetInfo,
    DataType,
    ModelInfo,
)

DATASET_FIDDLER_ID = '__fiddler_id'
LOG = logging.getLogger(__name__)


def _try_series_retype(series: pd.Series, new_type) -> pd.Series:
    try:
        return series.astype(new_type)
    except (TypeError, ValueError) as e:
        if new_type == 'int':
            LOG.warning(f'"{series.name}" cannot be loaded as int '
                        f'(likely because it contains missing values, and '
                        f'Pandas does not support NaN for ints). Loading '
                        f'as float instead.')
            return series.astype('float')
        else:
            raise e


def df_from_json_rows(dataset_rows_json: List[dict],
                      dataset_info: DatasetInfo,
                      include_fiddler_id: bool = False) -> pd.DataFrame:
    """Converts deserialized JSON into a pandas DataFrame according to a
        DatasetInfo object.

    If `include_fiddler_id` is true, we assume there is an extra column at
    the zeroth position containing the fiddler ID.
    """
    column_names = dataset_info.get_column_names()
    if include_fiddler_id:
        column_names.insert(0, DATASET_FIDDLER_ID)
    if include_fiddler_id:
        dataset_info = copy.deepcopy(dataset_info)
        dataset_info.columns.insert(
            0, Column(DATASET_FIDDLER_ID, DataType.INTEGER))
    df = pd.DataFrame(dataset_rows_json,
                      columns=dataset_info.get_column_names())
    for column_name in df:
        dtype = dataset_info[column_name].get_pandas_dtype()
        df[column_name] = _try_series_retype(df[column_name], dtype)
    return df


def retype_df_for_model(df: pd.DataFrame, model_info: ModelInfo)\
        -> pd.DataFrame:
    all_columns = (model_info.inputs if model_info.targets is None
                   else model_info.inputs + model_info.targets)
    for column in all_columns:
        if column.name in df:
            df[column.name] = _try_series_retype(
                df[column.name], column.get_pandas_dtype())
    return df


def _df_to_dict(df: pd.DataFrame):
    data_array = [y.iloc[0, :].to_dict() for x, y in df.groupby(level=0)]
    # convert numpy type values to python type: some numpy types are not JSON serializable
    for data in data_array:
        for key, val in data.items():
            if isinstance(val, np.bool_):
                data[key] = bool(val)
            if isinstance(val, np.int64):
                data[key] = int(val)
            if isinstance(val, np.float32):
                data[key] = float(val)
    return data_array


class ColorLogger:

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def _log(self, message, color):
        print(f"{color}{message}{self.WHITE}")

    def info(self, message):
        self._log(message, self.BLUE)

    def success(self, message):
        self._log(message, self.GREEN)

    def error(self, message):
        self._log(message, self.RED)

    def warn(self, message):
        self._log(message, self.YELLOW)
