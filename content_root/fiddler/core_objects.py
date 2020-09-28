# TODO: Add License

import copy
import enum
import functools
import json
import itertools
from pathlib import Path
import textwrap
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
import pandas.api.types
import numpy as np


def is_greater_than_max_value(value, limit, epsilon=1e-9):
    """ Check if 'value' is significantly larger than 'limit'.
    Where significantly means theres a greater difference than epsilon.
    """
    max_arg = abs(max(value, limit))
    tolerance = max_arg * epsilon
    # make sure that a is "significantly" smaller than b before declaring it
    # identified in DI for hired
    return value - tolerance > limit


def is_less_than_min_value(value, limit, epsilon=1e-9):
    """ Check if 'value' is significantly smaller than 'limit'.
    Where significantly means theres a greater difference than epsilon.
    """
    max_arg = abs(max(value, limit))
    tolerance = max_arg * epsilon
    # make sure that a is "significantly" smaller than b before declaring it
    # identified in DI for hired
    return value + tolerance < limit


class IntegrityViolationStatus(NamedTuple):
    is_nullable_violation: bool
    is_type_violation: bool
    is_range_violation: bool


@enum.unique
class DataType(enum.Enum):
    """Supported datatypes for the Fiddler engine."""

    FLOAT = 'float'
    INTEGER = 'int'
    BOOLEAN = 'bool'
    STRING = 'str'
    CATEGORY = 'category'

    def is_numeric(self):
        return self.value in (DataType.INTEGER.value, DataType.FLOAT.value)


@enum.unique
class ModelTask(enum.Enum):
    """Supported model tasks for the Fiddler engine."""

    BINARY_CLASSIFICATION = 'binary_classification'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    REGRESSION = 'regression'


@enum.unique
class ModelInputType(enum.Enum):
    """Supported model paradigms for the Fiddler engine."""

    TABULAR = 'structured'
    TEXT = 'text'


class AttributionExplanation(NamedTuple):
    """The results of an attribution explanation run by the Fiddler engine."""

    algorithm: str
    inputs: List[str]
    attributions: List[float]
    misc: Optional[dict]

    @classmethod
    def from_dict(cls, deserialized_json: dict):
        """Converts a deserialized JSON format into an
        AttributionExplanation object"""
        algorithm = deserialized_json.pop('explanation_type')
        inputs, attributions = zip(*deserialized_json.pop('explanation').items())
        return cls(
            algorithm=algorithm,
            inputs=list(inputs),
            attributions=list(attributions),
            misc=deserialized_json,
        )


class MulticlassAttributionExplanation(NamedTuple):
    """A collection of AttributionExplanation objects explaining several
    classes' predictions in a multiclass classification setting."""

    classes: Tuple[str]
    explanations: Dict[str, AttributionExplanation]

    @classmethod
    def from_dict(cls, deserialized_json: dict):
        """Converts a deserialized JSON format into an
        MulticlassAttributionExplanation object"""
        return cls(
            classes=tuple(deserialized_json.keys()),
            explanations={
                label_class: AttributionExplanation.from_dict(explanation_dict)
                for label_class, explanation_dict in deserialized_json.items()
            },
        )


class MLFlowParams:
    """Holds the configuration information for a model packaged as an MLFlow
        model."""

    def __init__(
        self,
        relative_path_to_saved_model: Union[str, Path],
        live_endpoint: Optional[str] = None,
    ):
        self.relative_path_to_saved_model = Path(relative_path_to_saved_model)
        self.live_endpoint = live_endpoint

    @classmethod
    def from_dict(cls, d):
        return cls(d['relative_path_to_saved_model'], d.get('live_endpoint', None))

    def to_dict(self):
        res = {
            'relative_path_to_saved_model': str(self.relative_path_to_saved_model),
        }
        if self.live_endpoint is not None:
            res['live_endpoint'] = self.live_endpoint
        return res


class Column:
    """Represents a single column of a dataset or model input/output.

    :param name: The name of the column (corresponds to the header row of a
        CSV file)
    :param data_type: The best encoding type for this column's data.
    :param possible_values: If data_type is CATEGORY, then an exhaustive list
        of possible values for this category must be provided. Otherwise
        this field has no effect and is optional.
    :param is_nullable: Optional metadata. Tracks whether or not this column is
        expected to contain some null values.
    :param value_range_x: Optional metadata. If data_type is FLOAT or INTEGER,
        then these values specify a range this column's values are expected to
        stay within. Has no effect for non-numerical data_types.
    """

    def __init__(
        self,
        name: str,
        data_type: DataType,
        possible_values: Optional[List[Any]] = None,
        is_nullable: Optional[bool] = None,
        value_range_min: Optional[float] = None,
        value_range_max: Optional[float] = None,
    ):
        self.name = name
        self.data_type = data_type
        self.possible_values = possible_values
        self.is_nullable = is_nullable
        self.value_range_min = value_range_min
        self.value_range_max = value_range_max

        if (
            self.data_type.value != DataType.CATEGORY.value
            and self.possible_values is not None
        ):
            raise ValueError(
                f'Do not pass `possible_values` for '
                f'non-categorical {self.data_type} data type.'
            )
        inappropriate_value_range = not self.data_type.is_numeric() and not (
            self.value_range_min is None and self.value_range_max is None
        )
        if inappropriate_value_range:
            raise ValueError(
                f'Do not pass `value_range` for '
                f'non-numerical {self.data_type} data type.'
            )

    @classmethod
    def from_dict(cls, desrialized_json: dict):
        """Creates a Column object from deserialized JSON"""
        return cls(
            name=desrialized_json['column-name'],
            data_type=DataType(desrialized_json['data-type']),
            possible_values=desrialized_json.get('possible-values', None),
            is_nullable=desrialized_json.get('is-nullable', None),
            value_range_min=desrialized_json.get('value-range-min', None),
            value_range_max=desrialized_json.get('value-range-max', None),
        )

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        res = (
            f'Column(name="{self.name}", data_type={self.data_type}, '
            f'possible_values={self.possible_values}'
        )
        if self.is_nullable is not None:
            res += f', is_nullable={self.is_nullable}'
        if self.value_range_min is not None or self.value_range_max is not None:
            res += (
                f', value_range_min={self.value_range_min}'
                f', value_range_max={self.value_range_max}'
            )
        res += ')'
        return res

    def _raise_on_bad_categorical(self):
        """Raises a ValueError if data_type=CATEGORY without possible_values"""
        if (
            self.data_type.value == DataType.CATEGORY.value
            and self.possible_values is None
        ):
            raise ValueError(
                f'Mal-formed categorical column missing `possible_values`: ' f'{self}'
            )

    def get_pandas_dtype(self):
        """Converts the data_type field to a Pandas-friendly form."""
        self._raise_on_bad_categorical()
        if self.data_type.value == DataType.CATEGORY.value:
            return pandas.api.types.CategoricalDtype(self.possible_values)
        else:
            return self.data_type.value

    def to_dict(self) -> Dict[str, Any]:
        """Converts this object to a more JSON-friendly form."""
        res = {
            'column-name': self.name,
            'data-type': self.data_type.value,
        }
        if self.possible_values is not None:
            # possible-values can be string, int, etc
            # no need to convert everything to str
            res['possible-values'] = [val for val in self.possible_values]
        if self.is_nullable is not None:
            res['is-nullable'] = self.is_nullable
        if self.value_range_min is not None:
            res['value-range-min'] = self.value_range_min
        if self.value_range_max is not None:
            res['value-range-max'] = self.value_range_max
        return res

    def violation_of_value(self, value):
        if Column._value_is_na_or_none(value):
            return False
        if self.data_type.is_numeric():
            is_too_low = self.value_range_min is not None and is_less_than_min_value(
                value, self.value_range_min
            )
            is_too_high = (
                self.value_range_max is not None
                and is_greater_than_max_value(value, self.value_range_max)
            )
            return is_too_low or is_too_high
        if self.data_type.value == DataType.CATEGORY.value:
            return value not in self.possible_values
        return False

    def violation_of_type(self, value):
        if Column._value_is_na_or_none(value):
            return False
        if self.data_type.value == DataType.FLOAT.value:
            # json loading from string reads non-decimal number always as int
            return not (isinstance(value, float) or isinstance(value, int))
        if self.data_type.value == DataType.INTEGER.value:
            return not isinstance(value, int)
        if self.data_type.value == DataType.STRING.value:
            return not isinstance(value, str)
        if self.data_type.value == DataType.BOOLEAN.value:
            return not isinstance(value, bool) and value not in (0, 1)
        if self.data_type.value == DataType.CATEGORY.value:
            possible_types = tuple(set(type(v) for v in self.possible_values))
            return not isinstance(value, possible_types)

    def violation_of_nullable(self, value):
        if self.is_nullable is not None and self.is_nullable is False:
            return Column._value_is_na_or_none(value)
        return False

    def check_violation(self, value):
        if self.violation_of_nullable(value):
            return IntegrityViolationStatus(True, False, False)
        if self.violation_of_type(value):
            return IntegrityViolationStatus(False, True, False)
        if self.violation_of_value(value):
            return IntegrityViolationStatus(False, False, True)
        return IntegrityViolationStatus(False, False, False)

    @staticmethod
    def _value_is_na_or_none(value):
        if value is None:
            return True
        try:
            return np.isnan(value)
        except TypeError:
            return False


def _get_field_pandas_dtypes(
    column_sequence: Sequence[Column],
) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
    """Get a dictionary describing the pandas datatype of every column in a
    sequence of columns."""
    dtypes = dict()
    for column in column_sequence:
        dtypes[column.name] = column.get_pandas_dtype()
    return dtypes


class DatasetInfo:
    """Information about a dataset. Defines the schema.

    :param display_name: A name for user-facing display (different from an id).
    :param columns: A list of Column objects.
    :param files: Optional. If the dataset is stored in one or more CSV files
        with canonical names, this field lists those files. Primarily for use
        only internally to the Fiddler engine.
    """

    def __init__(
        self,
        display_name: str,
        columns: List[Column],
        files: Optional[List[str]] = None,
        dataset_id: str = None,
        **kwargs,
    ):
        self.display_name = display_name
        self.dataset_id = dataset_id
        self.columns = columns
        self.files = files if files is not None else list()
        self.misc = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Converts this object to a more JSON-friendly form."""
        res = {
            'name': self.display_name,
            'columns': [c.to_dict() for c in self.columns],
            'files': self.files,
        }
        return {**res, **self.misc}

    def get_column_names(self) -> List[str]:
        """Returns a list of column names."""
        return [column.name for column in self.columns]

    def get_column_pandas_dtypes(
        self,
    ) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
        """Get a dictionary describing the pandas datatype of every column."""
        return _get_field_pandas_dtypes(self.columns)

    def get_event_integrity(
        self, event: Dict[str, Union[str, float, int, bool]]
    ) -> Tuple[IntegrityViolationStatus, ...]:
        if not set(event.keys()).issuperset(set(self.get_column_names())):
            raise ValueError(
                f'Event feature names {set(event.keys())} not'
                f'a superset of column names '
                f'{set(self.get_column_names())}'
            )
        return tuple(
            column.check_violation(event[column.name]) for column in self.columns
        )

    @staticmethod
    def _datatype_from_pandas_dtype(pandas_col) -> DataType:
        dtype_obj = pandas_col.dtype
        # catch uint8 one-hot encoded variables
        if dtype_obj.name == 'uint8' and pandas_col.nunique() < 3:
            return DataType.BOOLEAN
        if dtype_obj.kind == 'i':
            return DataType.INTEGER
        elif dtype_obj.kind == 'f':
            return DataType.FLOAT
        elif dtype_obj.kind == 'b':
            return DataType.BOOLEAN
        elif dtype_obj.kind and dtype_obj.name == 'category':
            return DataType.CATEGORY
        else:
            return DataType.STRING

    @classmethod
    def update_stats_for_existing_schema(
        cls,
        dataset: dict,
        info,
        max_inferred_cardinality: Optional[int] = None,
        zero_one_as_int: bool = False,
    ):
        """ Takes a customer/user provided schema along with a bunch
        of files with corresponding data in dataframes and merges them
        together and updates the user schema.
        Please note that we DO NOT update stats in the user provided
        schema if those stats are already there. We assume that the
        user wants those stats for data integrity testing.
        """
        updated_infos = []
        for name, item in dataset.items():
            update_info = DatasetInfo.check_and_update_column_info(info, item)
            updated_infos.append(update_info)
        info = DatasetInfo.as_combination(updated_infos, display_name=info.display_name)
        return info

    @classmethod
    def check_and_update_column_info(
        cls,
        info_original,
        df: pd.DataFrame,
        max_inferred_cardinality: Optional[int] = None,
        zero_one_as_int: bool = False,
    ):
        """ When called on a Dataset, this function will calculate stats
        that are used by DI and put add them to each Column in case its
        not already there. Currently stats include is_nullable, possible_values, and
        min/max ranges.
        Please note that we DO NOT update stats in the user provided
        schema if those stats are already there. We assume that the
        user wants those stats for data integrity testing.
        """

        info = copy.deepcopy(info_original)
        # add index column if it is not just an unnamed RangeIndex
        name_series_iter = df.items()
        if df.index.name is not None:
            name_series_iter = itertools.chain(
                (df.index.name, df.index.to_series()), name_series_iter
            )
        column_stats = {}
        for column_name, column_series in name_series_iter:
            column_info = cls._calculate_stats_for_col(
                column_name, column_series, max_inferred_cardinality, zero_one_as_int
            )
            column_stats[column_name] = column_info

        for column in info.columns:
            # Fill in stats for each column if its not present
            column_info = column_stats[column.name]
            if not column.is_nullable:
                column.is_nullable = column_info.is_nullable
            if not column.value_range_min:
                column.value_range_min = column_info.value_range_min
            if not column.value_range_max:
                column.value_range_max = column_info.value_range_max
            if not column.possible_values:
                column.possible_values = column_info.possible_values

        return cls(info.display_name, info.columns)

    @classmethod
    def from_dataframe(
        cls,
        df: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        display_name: str = '',
        max_inferred_cardinality: Optional[int] = None,
        zero_one_as_int: bool = False,
    ):
        """Infers a DatasetInfo object from a pandas DataFrame
        (or iterable of DataFrames).

        :param df: Either a single DataFrame or an iterable of DataFrame
            objects. If an iterable is given, all dataframes must have the
            same columns.
        :param display_name: A name for user-facing display (different from
            an id).
        :param max_inferred_cardinality: Optional. If not None, any
            string-typed column with fewer than `max_inferred_cardinality`
            unique values will be inferred as a category (useful for cases
            where use of the built-in CategoricalDtype functionality of Pandas
            is not desired).
        :param zero_one_as_int: bool. If True, skip inferring integer valued
            column of only {0, 1} values as type BOOLEAN.

        :returns: A DatasetInfo object.
        """
        # if an iterable is passed, infer for each in the iterable and combine
        if not isinstance(df, pd.DataFrame):
            info_gen = (
                cls.from_dataframe(
                    item, max_inferred_cardinality=max_inferred_cardinality
                )
                for item in df
            )
            return cls.as_combination(info_gen, display_name=display_name)

        columns = []
        name_series_iter = df.items()
        # add index column if it is not just an unnamed RangeIndex
        if df.index.name is not None:
            name_series_iter = itertools.chain(
                (df.index.name, df.index.to_series()), name_series_iter
            )
        for column_name, column_series in name_series_iter:
            column_info = cls._calculate_stats_for_col(
                column_name, column_series, max_inferred_cardinality, zero_one_as_int
            )
            columns.append(column_info)
        return cls(display_name, columns)

    @staticmethod
    def _calculate_stats_for_col(
        column_name, column_series, max_inferred_cardinality, zero_one_as_int
    ):
        column_dtype = DatasetInfo._datatype_from_pandas_dtype(column_series)

        # infer categorical if configured to do so
        if (
            max_inferred_cardinality is not None
            and column_dtype.value == DataType.STRING.value
            and (column_series.nunique() < max_inferred_cardinality)
        ):
            column_dtype = DataType.CATEGORY

        # check for boolean as int
        if (
            not zero_one_as_int
            and column_dtype.value == DataType.INTEGER.value
            and not column_series.isna().any()
            and set(column_series.unique()).issubset({0, 1})
        ):
            column_dtype = DataType.BOOLEAN

        # get possible values for categorical type
        if column_dtype.value == DataType.CATEGORY.value:
            unique = column_series.unique()
            # we have to check for NaN VERY CAREFULLY,
            #  since np.isnan won't work on object-type arrays
            nan_idx = None
            for i, val in enumerate(unique):
                if val is np.nan:
                    nan_idx = i
                    break
            possible_values = list(map(str, unique.tolist()))
            if nan_idx is not None:
                possible_values.pop(nan_idx)
            possible_values = sorted(possible_values)
        else:
            possible_values = None

        # get value range for numerical dtype
        if column_dtype.is_numeric():
            value_min, value_max = column_series.agg(['min', 'max'])
            if np.isnan(value_min):
                value_min = None
            if np.isnan(value_max):
                value_max = None
        else:
            value_min, value_max = None, None

        # get nullability
        is_nullable = bool(column_series.isna().any())
        return Column(
            name=column_name,
            data_type=column_dtype,
            possible_values=possible_values,
            is_nullable=is_nullable,
            value_range_min=value_min,
            value_range_max=value_max,
        )

    @classmethod
    def from_dict(cls, deserialized_json: dict):
        """Transforms deserialized JSON into a DatasetInfo object"""
        # drop down into the "dataset" object inside the deserialized_json
        deserialized_json = deserialized_json['dataset']

        # instantiate the class
        return cls(
            display_name=deserialized_json['name'],
            columns=[Column.from_dict(c) for c in deserialized_json['columns']],
            files=deserialized_json.get('files', None),
        )

    @classmethod
    def _combine(cls, info_a, info_b, display_name: str = ''):
        """Given two DatasetInfo objects, tries to combine them into
            a single DatasetInfo that describes both sub-datasets."""
        # raise error if column names are incompatible
        if info_a.get_column_names() != info_b.get_column_names():
            raise ValueError(
                f'Incompatible DatasetInfo objects: column names do not '
                f'match:\n{info_a.get_column_names()}\n'
                f'{info_b.get_column_names()}'
            )

        # combine columns
        columns = list()
        for a_column, b_column in zip(info_a.columns, info_b.columns):
            # resolve types
            a_type, b_type = a_column.data_type.value, b_column.data_type.value
            if a_type == b_type:
                col_type = a_column.data_type
            elif {a_type, b_type}.issubset(
                {DataType.BOOLEAN.value, DataType.INTEGER.value}
            ):
                col_type = DataType.INTEGER
            elif {a_type, b_type}.issubset(
                {DataType.BOOLEAN.value, DataType.INTEGER.value, DataType.FLOAT.value}
            ):
                col_type = DataType.FLOAT
            else:
                col_type = DataType.STRING

            # resolve possible_values
            if col_type.value == DataType.CATEGORY.value:
                assert a_column.possible_values is not None
                assert b_column.possible_values is not None
                possible_values = list(
                    set(a_column.possible_values) | set(b_column.possible_values)
                )
            else:
                possible_values = None

            # resolve is_nullable, priority being True, then False, then None
            if a_column.is_nullable is None and b_column.is_nullable is None:
                is_nullable = None
            elif a_column.is_nullable or b_column.is_nullable:
                is_nullable = True
            else:
                is_nullable = False

            # resolve value range
            value_range_min = a_column.value_range_min
            if b_column.value_range_min is not None:
                if value_range_min is None:
                    value_range_min = b_column.value_range_min
                else:
                    value_range_min = min(value_range_min, b_column.value_range_min)
            value_range_max = a_column.value_range_max
            if b_column.value_range_max is not None:
                if value_range_max is None:
                    value_range_max = b_column.value_range_max
                else:
                    value_range_max = max(value_range_max, b_column.value_range_max)
            columns.append(
                Column(
                    a_column.name,
                    col_type,
                    possible_values,
                    is_nullable,
                    value_range_min,
                    value_range_max,
                )
            )

        # combine file lists
        files = info_a.files + info_b.files

        return cls(display_name, columns, files)

    @classmethod
    def as_combination(cls, infos: Iterable, display_name: str = 'combined_dataset'):
        """Combines an iterable of compatible DatasetInfo objects into one
        DatasetInfo"""
        return functools.reduce(
            lambda a, b: cls._combine(a, b, display_name=display_name), infos
        )

    @staticmethod
    def get_summary_dataframe(dataset_info):
        """Returns a table (pandas DataFrame) summarizing the DatasetInfo."""
        return _summary_dataframe_for_columns(dataset_info.columns)

    def __repr__(self):
        column_info = textwrap.indent(repr(self.get_summary_dataframe(self)), '    ')
        return (
            f'DatasetInfo:\n'
            f'  display_name: {self.display_name}\n'
            f'  files: {self.files}\n'
            f'  columns:\n'
            f'{column_info}'
        )

    def _col_id_from_name(self, name):
        """Look up the index of the column by name"""
        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        raise KeyError(name)

    def __getitem__(self, item):
        return self.columns[self._col_id_from_name(item)]

    def __setitem__(self, key, value):
        assert isinstance(value, Column), 'Must set column to be a ' '`Column` object'
        self.columns[self._col_id_from_name(key)] = value

    def __delitem__(self, key):
        del self.columns[self._col_id_from_name(key)]


class ModelInfo:
    """Information about a model. Stored in `model.yaml` file on the backend.

    :param display_name: A name for user-facing display (different from an id).
    :param input_type: Specifies whether the model is in the tabular or text
        paradigm.
    :param model_task: Specifies the task the model is designed to address.
    :param inputs: A list of Column objects corresponding to the dataset
        columns that are fed as inputs into the model.
    :param outputs: A list of Column objects corresponding to the table
        output by the model when running predictions.
    :param targets: A list of Column objects corresponding to the dataset
        columns used as targets/labels for the model. If not provided, some
        functionality (like scoring) will not be available.
    :param framework: A string providing information about the software library
        and version used to train and run this model.
    :param description: A user-facing description of the model.
    :param mlflow_params: MLFlow parameters.
    :param **kwargs: Additional information about the model to store as `misc`.
    """

    def __init__(
        self,
        display_name: str,
        input_type: ModelInputType,
        model_task: ModelTask,
        inputs: List[Column],
        outputs: List[Column],
        metadata: Optional[List[Column]] = None,
        decisions: Optional[List[Column]] = None,
        targets: Optional[List[Column]] = None,
        framework: Optional[str] = None,
        description: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        mlflow_params: Optional[MLFlowParams] = None,
        **kwargs,
    ):
        self.display_name = display_name
        self.input_type = input_type
        self.model_task = model_task
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.metadata = metadata
        self.decisions = decisions
        self.framework = framework
        self.description = description
        self.datasets = datasets
        self.mlflow_params = mlflow_params
        self.misc = kwargs

    def to_dict(self):
        """Dumps to basic python objects (easy for JSON serialization)"""
        res = {
            'name': self.display_name,
            'input-type': self.input_type.value,
            'model-task': self.model_task.value,
            'inputs': [c.to_dict() for c in self.inputs],
            'outputs': [c.to_dict() for c in self.outputs],
            'datasets': self.datasets or [],
        }
        if self.metadata is not None:
            res['metadata'] = [metadata_col.to_dict() for metadata_col in self.metadata]
        if self.decisions is not None:
            res['decisions'] = [outcome_col.to_dict() for outcome_col in self.decisions]
        if self.targets is not None:
            res['targets'] = [target_col.to_dict() for target_col in self.targets]
        if self.description is not None:
            res['description'] = self.description
        if self.framework is not None:
            res['framework'] = self.framework
        if self.mlflow_params is not None:
            res['mlflow'] = self.mlflow_params.to_dict()
        return {**res, **self.misc}

    def get_input_names(self):
        """Returns a list of names for model inputs."""
        return [column.name for column in self.inputs]

    def get_output_names(self):
        """Returns a list of names for model outputs."""
        return [column.name for column in self.outputs]

    def get_metadata_names(self):
        """Returns a list of names for model metadata."""
        return [column.name for column in self.metadata]

    def get_outcome_names(self):
        """Returns a list of names for model decisions."""
        return [column.name for column in self.decisions]

    def get_target_names(self):
        """Returns a list of names for model targets."""
        return [column.name for column in self.targets]

    def get_input_pandas_dtypes(
        self,
    ) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
        """Get a dictionary describing the pandas datatype of every input."""
        return _get_field_pandas_dtypes(self.inputs)

    def get_output_pandas_dtypes(
        self,
    ) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
        """Get a dictionary describing the pandas datatype of every output."""
        return _get_field_pandas_dtypes(self.outputs)

    def get_metadata_pandas_dtypes(
        self,
    ) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
        """Get a dictionary describing the pandas datatype of every
        metadata column."""
        return _get_field_pandas_dtypes(self.metadata)

    def get_decisions_pandas_dtypes(
        self,
    ) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
        """Get a dictionary describing the pandas datatype of every outcome
        column. """
        return _get_field_pandas_dtypes(self.decisions)

    def get_target_pandas_dtypes(
        self,
    ) -> Dict[str, Union[str, pandas.api.types.CategoricalDtype]]:
        """Get a dictionary describing the pandas datatype of every target."""
        return _get_field_pandas_dtypes(self.targets)

    @classmethod
    def from_dict(cls, deserialized_json: dict):
        """Transforms deserialized JSON into a ModelInfo object"""
        # drop down into the "model" object inside the deserialized_json
        # (work on a copy)
        deserialized_json = copy.deepcopy(deserialized_json['model'])

        name = deserialized_json.pop('name')
        input_type = ModelInputType(deserialized_json.pop('input-type'))
        model_task = ModelTask(deserialized_json.pop('model-task'))
        inputs = [Column.from_dict(c) for c in deserialized_json.pop('inputs')]
        outputs = [Column.from_dict(c) for c in deserialized_json.pop('outputs')]
        if 'metadata' in deserialized_json:
            metadata = [Column.from_dict(c) for c in deserialized_json.pop('metadata')]
        else:
            metadata = None

        if 'decisions' in deserialized_json:
            decisions = [
                Column.from_dict(c) for c in deserialized_json.pop('decisions')
            ]
        else:
            decisions = None

        if 'targets' in deserialized_json:
            targets = [Column.from_dict(c) for c in deserialized_json.pop('targets')]
        else:
            targets = None
        description = deserialized_json.pop('description', None)
        if 'mlflow' in deserialized_json:
            mlflow_params = MLFlowParams.from_dict(deserialized_json['mlflow'])
        else:
            mlflow_params = None

        if 'datasets' in deserialized_json:
            datasets = deserialized_json.pop('datasets')
        else:
            datasets = None

        # instantiate the class
        return cls(
            display_name=name,
            input_type=input_type,
            model_task=model_task,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            decisions=decisions,
            targets=targets,
            description=description,
            datasets=datasets,
            mlflow_params=mlflow_params,
            **deserialized_json,
        )

    @classmethod
    def from_dataset_info(
        cls,
        dataset_info: DatasetInfo,
        target: str,
        features: Optional[Sequence[str]] = None,
        metadata_cols: Optional[Sequence[str]] = None,
        outcome_cols: Optional[Sequence[str]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        input_type: ModelInputType = ModelInputType.TABULAR,
        model_task: Optional[ModelTask] = None,
        mlflow_params: Optional[MLFlowParams] = None,
    ):
        """Produces a ModelInfo for a model trained on a dataset.

        :param dataset_info: A DatasetInfo object describing the training
            dataset.
        :param target: The column name of the target the model predicts.
        :param features: A list of column names for columns used as features.
        :param metadata_cols: A list of column names for columns used as
        metadata.
        :param outcome_cols: A list of column names for columns used as
        decisions.
        :param display_name: A model name for user-facing display (different
            from an id).
        :param description: A user-facing description of the model.
        :param input_type: Specifies the paradigm (tabular or text) of the
            model.
        :param model_task: Specifies the prediction task addressed by the
            model. If not explicitly provided, this will be inferred from the
            data type of the target variable.
        :param mlflow_params: MLFlow parameters.

        :returns A ModelInfo object.
        """
        if display_name is None:
            display_name = f'{dataset_info.display_name} model'

        # infer inputs, and add metadata and outcome columns, if they exist

        inputs = list()

        additional_columns = list()
        if metadata_cols is not None:
            additional_columns += metadata_cols
            metadata = list()
        else:
            metadata = None

        if outcome_cols is not None:
            additional_columns += outcome_cols
            decisions = list()
        else:
            decisions = None

        # ensure that columns are not duplicated
        if len(additional_columns) > 0:
            col_list = [target]
            if features is not None:
                col_list += features

            duplicated_cols = [col for col in additional_columns if col in col_list]

            if len(duplicated_cols) > 0:
                raise ValueError(
                    f'Cols can be either feature, target, '
                    f'metadata or decisions. Cols '
                    f'{",".join(duplicated_cols)} are present '
                    f'in more than one category'
                )

        for column in dataset_info.columns:
            col_name = column.name
            if (
                col_name != target
                and (features is None or col_name in features)
                and (col_name not in additional_columns)
            ):
                inputs.append(column.copy())
            if metadata_cols and col_name in metadata_cols:
                metadata.append(column.copy())
            if outcome_cols and col_name in outcome_cols:
                decisions.append(column.copy())

        # determine target column
        try:
            target_column = dataset_info[target]
        except KeyError:
            raise ValueError(f'Target "{target}" not found in dataset.')

        # infer task type, outputs, and target levels from target
        if target_column.data_type.value == DataType.BOOLEAN.value:
            target_levels = [False, True]
            outputs = [
                Column(
                    name=f'probability_{target_column.name}_True',
                    data_type=DataType.FLOAT,
                    is_nullable=False,
                    value_range_min=0.0,
                    value_range_max=1.0,
                )
            ]
            if model_task is None:
                model_task = ModelTask.BINARY_CLASSIFICATION
        elif target_column.data_type.value == DataType.CATEGORY.value:
            target_levels = target_column.possible_values
            if model_task is None:
                if len(target_levels) == 2:
                    model_task = ModelTask.BINARY_CLASSIFICATION
                else:
                    model_task = ModelTask.MULTICLASS_CLASSIFICATION
            if model_task.value == ModelTask.BINARY_CLASSIFICATION.value:
                outputs = [
                    Column(
                        name=f'probability_{target_levels[1]}',
                        data_type=DataType.FLOAT,
                        is_nullable=False,
                        value_range_min=0.0,
                        value_range_max=1.0,
                    )
                ]
            else:
                outputs = [
                    Column(
                        name=f'probability_{level}',
                        data_type=DataType.FLOAT,
                        is_nullable=False,
                        value_range_min=0.0,
                        value_range_max=1.0,
                    )
                    for level in target_levels
                ]
        else:
            target_levels = None
            outputs = [
                Column(
                    name=f'predicted_{target_column.name}',
                    data_type=DataType.FLOAT,
                    is_nullable=False,
                )
            ]
            if model_task is None:
                model_task = ModelTask.REGRESSION

        datasets = None
        if dataset_info.dataset_id is not None:
            datasets = [dataset_info.dataset_id]

        return cls(
            display_name=display_name,
            description=description,
            input_type=input_type,
            model_task=model_task,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            decisions=decisions,
            datasets=datasets,
            targets=[target_column],
            mlflow_params=mlflow_params,
        )

    @staticmethod
    def get_summary_dataframes_dict(model_info):
        """Returns a dictionary of DataFrames summarizing the
        ModelInfo's inputs, outputs, and if they exist, metadata
        and decisions"""

        summary_dict = dict()

        summary_dict['inputs'] = _summary_dataframe_for_columns(model_info.inputs)

        summary_dict['outputs'] = _summary_dataframe_for_columns(model_info.outputs)

        if model_info.metadata:
            summary_dict['metadata'] = _summary_dataframe_for_columns(
                model_info.metadata
            )
        if model_info.decisions:
            summary_dict['decisions'] = _summary_dataframe_for_columns(
                model_info.decisions
            )

        return summary_dict

    def __repr__(self):
        summary_dict = ModelInfo.get_summary_dataframes_dict(self)
        input_info = textwrap.indent(repr(summary_dict['inputs']), '    ')
        output_info = textwrap.indent(repr(summary_dict['outputs']), '    ')

        metadata_info = (
            f'  metadata:\n'
            f"{textwrap.indent(repr(summary_dict['metadata']),  '    ')}"
            if self.metadata is not None
            else ''
        )

        decisions_info = (
            f'  decisions:\n'
            f"{textwrap.indent(repr(summary_dict['decisions']),  '    ')}"
            if self.decisions is not None
            else ''
        )

        target_info = f'  targets: {self.targets}\n' if self.targets is not None else ''
        framework_info = (
            f'  framework: {self.framework}\n' if self.framework is not None else ''
        )
        misc_info = textwrap.indent(json.dumps(self.misc, indent=2), '    ')
        return (
            f'ModelInfo:\n'
            f'  display_name: {self.display_name}\n'
            f'  description: {self.description}\n'
            f'  input_type: {self.input_type}\n'
            f'  model_task: {self.model_task}\n'
            f'  inputs:\n'
            f'{input_info}\n'
            f'  outputs:\n'
            f'{output_info}\n'
            f'{metadata_info}\n'
            f'{decisions_info}\n'
            f'{target_info}'
            f'{framework_info}'
            f'  misc:\n'
            f'{misc_info}'
        )


def _summary_dataframe_for_columns(
    columns: Sequence[Column], placeholder=''
) -> pd.DataFrame:
    """
    Example:
             column     dtype count(possible_values) is_nullable            value_range
0       CreditScore   INTEGER                              False        376 - 850
1         Geography  CATEGORY                      3       False
2            Gender  CATEGORY                      2       False
3               Age   INTEGER                              False         18 - 82
4            Tenure   INTEGER                              False          0 - 10
5           Balance     FLOAT                              False        0.0 - 213,100.0
6     NumOfProducts   INTEGER                              False          1 - 4
7         HasCrCard  CATEGORY                      2       False
8    IsActiveMember  CATEGORY                      2       False
9   EstimatedSalary     FLOAT                              False      371.1 - 199,700.0
10          Churned  CATEGORY                      2       False

    """  # noqa E501
    column_names = []
    column_dtypes = []
    n_possible_values = []
    is_nullable = []
    mins, maxes = [], []
    for column in columns:
        column_names.append(column.name)
        column_dtypes.append(column.data_type.name)
        n_possible_values.append(
            len(column.possible_values)
            if column.possible_values is not None
            else placeholder
        )
        is_nullable.append(
            str(column.is_nullable) if column.is_nullable is not None else placeholder
        )
        if not column.data_type.is_numeric():
            mins.append(None)
            maxes.append(None)
        else:
            min_str = (
                _prettyprint_number(column.value_range_min)
                if column.value_range_min is not None
                else '*'
            )
            max_str = (
                _prettyprint_number(column.value_range_max)
                if column.value_range_max is not None
                else '*'
            )
            mins.append(min_str)
            maxes.append(max_str)
    range_pad_len = max(len(x) if x is not None else 0 for x in mins + maxes)
    value_range = [
        (placeholder if x is None else f'{x:>{range_pad_len}} - {y:<{range_pad_len}}')
        for x, y in zip(mins, maxes)
    ]
    return pd.DataFrame(
        {
            'column': column_names,
            'dtype': column_dtypes,
            'count(possible_values)': n_possible_values,
            'is_nullable': is_nullable,
            'value_range': value_range,
        }
    )


def _prettyprint_number(number, n_significant_figures=4):
    n_digits = len(f'{number:.0f}')
    return f'{round(number, n_significant_figures - n_digits):,}'
