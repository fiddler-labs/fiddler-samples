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
from .validator import PackageValidator, ValidationModule, ValidationChainSettings
from .core_objects import (
    Column,
    DatasetInfo,
    DataType,
    MLFlowParams,
    ModelInfo,
    ModelInputType,
    ModelTask
)
from .utils import ColorLogger

__all__ = [
    '__version__',
    'Column',
    'ColorLogger',
    'DatasetInfo',
    'DataType',
    'Fiddler',
    'FiddlerApi',
    'MLFlowParams',
    'ModelInfo',
    'ModelInputType',
    'ModelTask',
    'PredictionEventBundle',
    'PackageValidator',
    'ValidationChainSettings',
    'ValidationModule'
    'utils',
]
