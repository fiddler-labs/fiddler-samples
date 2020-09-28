"""
Fiddler Client Module
=====================

A Python client for Fiddler service.

TODO: Add Licence.
"""

from . import utils
from ._version import __version__
from .fiddler_api import FiddlerApi
from .client import Fiddler
from .client import PredictionEventBundle
from .core_objects import (
    Column,
    DatasetInfo,
    DataType,
    MLFlowParams,
    ModelInfo,
    ModelInputType,
    ModelTask,
)

__all__ = [
    '__version__',
    'Column',
    'DatasetInfo',
    'DataType',
    'Fiddler',
    'FiddlerApi',
    'MLFlowParams',
    'ModelInfo',
    'ModelInputType',
    'ModelTask',
    'PredictionEventBundle',
    'utils',
]
