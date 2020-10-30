from typing import List, Callable, NamedTuple, Tuple
import os
import importlib.util
import pandas as pd
import pathlib
import pickle


from .core_objects import ModelInfo, DatasetInfo, ModelTask, DataType
from .assets.pg_reserved_words import pg_reserved_words
from .utils import ColorLogger

PG_COLUMN_NAME_MAX_LENGTH = 63

# Validation core objects


class ValidationError:
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class ValidationChainSettings(NamedTuple):
    include_default_modules: bool = True
    present_errors: bool = True
    fail_on_first_error_set: bool = False

    @classmethod
    def from_dict(cls, settings: dict):
        """Creates a ValidationChainSettings instance based on a dict

        Args:
            settings (dict): dictionary with appropriate key-value pairs

        Returns:
            ValidationChainSettings
        """
        validation_settings = cls()
        allowed_keys = dict(validation_settings._asdict())  # OrderedDict to dict
        finalized_settings_dict = {}
        for key, value in settings.items():
            if key in allowed_keys:
                finalized_settings_dict[key] = value
        return validation_settings._replace(**finalized_settings_dict)


# Validation functionality classes


class ValidationModule:

    __module_name__ = "Validator"  # modules should replace this name

    def __init__(
        self, model: ModelInfo, dataset: DatasetInfo, assets_dir: pathlib.Path
    ):
        self.model = model
        self.dataset = dataset
        self.assets_dir = assets_dir
        self.errors = []  # List[ValidationError]

    def run(self):
        """Override this function, here for type safety
        """
        pass

    def get_package_module(self):
        try:
            spec = importlib.util.spec_from_file_location(
                "*", str(self.assets_dir) + "/package.py"
            )
            package_py = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(package_py)
            return package_py
        except Exception as e:
            raise e

    def create_mock_df(self):
        df_dict = {}
        for column in self.model.inputs:
            data = None
            if column.data_type is DataType.INTEGER:
                data = 5
            elif column.data_type is DataType.FLOAT:
                data = 5.5
            elif column.data_type is DataType.BOOLEAN:
                data = True
            elif column.data_type is DataType.STRING:
                data = "mock_value"
            else:
                # DataType.CATEGORY
                data = column.possible_values[0]
            df_dict[column.name] = [data]
        return pd.DataFrame(df_dict)

    def append_error(self, message: str):
        """Enforces strict typing for errors generated from the module

        Args:
            message (str)
        """
        self.errors.append(ValidationError(message))

    def reset_errors(self):
        self.errors = []


class PackageValidator:
    def __init__(
        self,
        model: ModelInfo,
        dataset: DatasetInfo,
        assets_dir: pathlib.Path,
        settings: ValidationChainSettings = ValidationChainSettings(),
    ):
        self.model = model
        self.dataset = dataset
        self.assets_dir = assets_dir
        self.settings = settings
        self.validation_chain = []

        self.logger = ColorLogger()

        if self.settings.include_default_modules:
            self._push_default_modules_to_chain()

    @classmethod
    def from_validation_chain(
        cls,
        model: ModelInfo,
        dataset: DatasetInfo,
        assets_dir: pathlib.Path,
        chain_file: str="chain.pkl",
        settings: ValidationChainSettings = ValidationChainSettings(),
    ):
        """Reads a pickle file with a validation chain from the given assets directory and creates a class instance

        Args:
            model (ModelInfo)
            dataset (DatasetInfo)
            assets_dir (pathlib.Path)
            chain_file (str, optional): name of pickled validation chain file. Defaults to "chain.pkl".
            settings (ValidationChainSettings, optional). Defaults to ValidationChainSettings().

        Returns:
            PackageValidator
        """
        chain_loc = assets_dir / chain_file
        str_loc = os.fspath(chain_loc)
        if not chain_loc.is_file():
            raise RuntimeError(f"Validation chain binary '{str_loc}' is not valid")
        with open(str_loc, 'rb') as pkl:
            class_defns_chain = pickle.load(pkl)
            validator = cls(model, dataset, assets_dir, settings)
            validator.validation_chain = [module(model, dataset, assets_dir) for module in class_defns_chain]
            return validator


    @classmethod
    def from_hosted_validation_chain(
        cls,
        model: ModelInfo,
        dataset: DatasetInfo,
        assets_dir: pathlib.Path,
        chain_url: str,
        settings: ValidationChainSettings = ValidationChainSettings(),
    ):
        """Loads a url hosted pickle file with a validation chain

        Args:
            model (ModelInfo)
            dataset (DatasetInfo)
            assets_dir (pathlib.Path)
            chain_url (str): location of validation chain binary
            settings (ValidationChainSettings, optional). Defaults to ValidationChainSettings().

        Raises:
            e: exceptio thrown when trying to load the binary file

        Returns:
            PackageValidator
        """
        try:
            chain = pickle.load(urllib.request.urlopen(url))
            validator = cls(model, dataset, assets_dir, settings)
            validator.validation_chain = [module(model, dataset, assets_dir) for module in class_defns_chain]
            return validator
        except Exception as e:
            raise e
        

    def _present_errors(self, errors: dict):
        """Presents errors per module in a user friendly manner
        ex.
            [Module 1]
                - Error A
                - Error B
            [Module 2]
                - Error C
                - Error D

        Args:
            errors (dict): errors returned from run_chain()
        """
        for module in errors:
            if module != "all_errors" and len(errors[module]) > 0:
                print(f"[{module}]")
                for error in errors[module]:
                    str_err = str(error)
                    self.logger.error(f"\t- {str_err}")

    def _push_default_modules_to_chain(self):
        for module in [DatasetValidator, ModelValidator]:
            self.push_module_to_chain(module)

    def run_chain(self) -> Tuple[bool, dict]:
        """Runs validation chain sequentially by module order

        Returns:
            Tuple[bool, List[ValidationError]]: success boolean, list of errors from modules
        """

        def fail(errors):
            print("Validation Result:", end=" ")
            self.logger.error("FAIL")
            return False, errors

        def success():
            print("Validation Result:", end=" ")
            self.logger.success("PASS")
            return True, errors

        errors = {"all_errors": []}
        for module in self.validation_chain:
            module.reset_errors()
            module.run()
            errors["all_errors"].extend(module.errors)
            errors[module.name] = module.errors
            if self.settings.fail_on_first_error_set and len(errors["all_errors"]):
                return fail(errors)
        if len(errors["all_errors"]):
            if self.settings.present_errors:
                self._present_errors(errors)
            return fail(errors)
        else:
            return success()

    def push_module_to_chain(self, module: ValidationModule, position: int = -1):
        """Adds a validation step to the chain in the form of a ValidationModule

        Args:
            module (ValidationModule)
            position (int, optional): optional position in the chain. Defaults to None.
        """
        if position < len(self.validation_chain) - 1:
            self.validation_chain.append(
                module(self.model, self.dataset, self.assets_dir)
            )
        else:
            self.validation_chain.insert(
                position, module(self.model, self.dataset, self.assets_dir)
            )

    def export_validation_chain(self, filename: str = "chain.pkl"):
        """Exports the validation chain as a list of class definitions in a pickle file

        Args:
            filename (str, optional): name of exported file. Defaults to "chain.pkl".
        """
        raw_modules = [module.__class__ for module in self.validation_chain]
        pkl_path = os.path.join(self.assets_dir, filename)
        with open(pkl_path, "wb+") as pkl_file:
            pickle.dump(raw_modules, pkl_file)


class DatasetValidator(ValidationModule):

    name = "Dataset"

    def _check_reserved_words(self):
        """Checks if any columns in dataset are named with PostgreSQL reserved words
        """
        for col in self.dataset.columns:
            if col.name.upper() in pg_reserved_words:
                self.append_error(
                    f"Reserved word '{col.name}' used as dataset column name"
                )

    def _check_columns_name_length(self):
        """Validates that no dataset columns have a name with a length over the PostgreSQL allowed maximum
        """
        for col in self.dataset.columns:
            if len(col.name) > PG_COLUMN_NAME_MAX_LENGTH:
                self.append_error(
                    f"Dataset column name {col.name} exceeds {PG_COLUMN_NAME_MAX_LENGTH} character maximum"
                )

    def _check_dataset_name_case(self):
        """Validates that a dataset is named with strictly lowercase letters
        """
        if self.dataset.dataset_id != None and not self.dataset.dataset_id.islower():
            self.append_error(
                f"Dataset name '{self.dataset.dataset_id}' may not contain uppercase letters"
            )

    def _check_dataset_name_spaces(self):
        """Checks if name has any whitespaces
        """
        if self.dataset.dataset_id != None:
            for c in self.dataset.dataset_id:
                if c.isspace():
                    self.append_error(
                        f"Dataset name '{self.dataset.dataset_id}' may not contain whitespaces"
                    )
                    break

    def run(self):
        self._check_reserved_words()
        self._check_columns_name_length()
        self._check_dataset_name_case()
        self._check_dataset_name_spaces()


class ModelValidator(ValidationModule):

    name = "Model"
    package_valid = True

    def _run_model_prediction(self) -> pd.DataFrame:
        """Uses the package.py interface to run a prediciton, returns DF

        Returns:
            pd.DataFrame: prediction
        """
        package_module = self.get_package_module()
        # Insert test for get_model, then for predict
        model = package_module.get_model()
        row = self.create_mock_df()
        output = model.predict(row)
        return output

    def _check_package_module_valid(self):
        package_module = self.get_package_module()
        if not hasattr(package_module, "get_model"):
            self.append_error(
                "Module package.py does not contain mandatory method 'get_model'"
            )
            self.package_valid = False
        else:
            model = package_module.get_model()
            if not hasattr(model, "predict"):
                self.append_error(
                    "Model returned from 'get_model' does not have mandatory attribute 'predict'"
                )
                self.package_valid = False

    def _check_outputs_match(self):
        """Validates that outputs listed in ModelInfo matches outputs in package.py
        """
        prediciton = self._run_model_prediction()
        prediciton_cols = [str(col) for col in prediciton.columns]
        model_output_cols = [col.name for col in self.model.outputs]
        if not set(prediciton_cols) == set(model_output_cols):
            self.append_error(
                f"Listed prediction outputs {prediciton_cols} and {model_output_cols} do not match"
            )

    def _check_model_dataset_case(self):
        """Validates that all datasets associated with the ModelInfo only have lowercase letters in their name
        """
        for dataset in self.model.datasets:
            if not dataset.islower():
                self.append_error(
                    f"Associated dataset '{dataset}' may not contain uppercase letters"
                )

    def _check_classification_type_output_count(self):
        """Validates that classification type matches number of outputs
        """
        task = self.model.model_task
        outputs = self.model.outputs
        if task == ModelTask.BINARY_CLASSIFICATION and len(outputs) > 1:
            self.append_error(
                "Binary classification task specified with more than one output column"
            )
        elif task == ModelTask.MULTICLASS_CLASSIFICATION and len(outputs) < 2:
            self.append_error(
                "Multiclass classification task specified with less than two output columns"
            )

    def run(self):
        self._check_package_module_valid()
        self._check_classification_type_output_count()
        self._check_model_dataset_case()
        if self.package_valid:
            self._check_outputs_match()
