"""Custom feature transformation implemented in Python!"""
import pandas as pd
import pandas.api.types
import sklearn.preprocessing


class CustomFeatureTransformer:
    def __init__(self, missing='sentinel', sentinel_value=-999,
                 missing_drop_threshold=0.2,
                 binarize_threshold=None, categorical_trim=None):
        """Complex transformer that takes in raw data and performs
        a number of adjustments:

        1. Binarize continuous values that are overwhelmingly zero.
        If `binarize_threshold` is specified, all missing values with
        more than that threshold proportion of values equal to zero will
        be converted into boolean varaibles recording whether
        or not they are nonzero.

        2. Replaces missing values
            A. If `missing` is 'sentinel' then all missing values in
            continuous columns will be replaced by the value specified
            by `sentinel_value`
            B. If `missing` is 'impute' then fields with more than
            `missing_drop_threshold` proportion of missing values will
            be dropped and the rest will have their missing values imputed
            to the mean.

        3. Continuous variables are rescaled with standard scaling.

        4. Categorical variables are "trimmed". If a dictionary mapping
        variable name to a threshold between zero and one is passed into the
        `categorical_trim` parameter, levels of each variable specified that do
        not account for at least the threshold proportion of the overall
        distribution of values will be removed from the variable and replaced
        with NaNs to prevent overfitting to these rare examples.

        5. Categorical variables are one-hot encoded. Missing values are
        counted as a distinct level.
        """
        if missing not in ('sentinel', 'impute'):
            raise ValueError('"missing" must either be '
                             '"sentinel" or "impute"')
        self.missing = missing
        self.sentinel_value = sentinel_value
        self.missing_drop_threshold = missing_drop_threshold
        self.binarize_threshold = binarize_threshold
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.binarized_fields = None
        self.categorical_trim = categorical_trim
        self.adjusted_categoricals = {}
        self.is_fit = False
        self.continuous_cols_in_order = None

    def fit(self, df):
        # first, binarize features
        continuous_df = df.select_dtypes(['float', 'int'])
        if self.binarize_threshold is not None:
            self.binarized_fields = [name
                                     for name, values
                                     in continuous_df.iteritems()
                                     if (values.eq(0).mean()
                                         > self.binarize_threshold)]
            continuous_df = continuous_df.drop(columns=self.binarized_fields)

        # then rescale the rest
        if self.missing == 'impute':
            should_drop = (continuous_df.isna().mean()
                           > self.missing_drop_threshold)
            drop_features = continuous_df.columns[should_drop]
            continuous_df = continuous_df.drop(columns=drop_features)
        self.continuous_cols_in_order = continuous_df.columns.tolist()
        self.scaler.fit(continuous_df)

        # lastly learn how to clip the categorical variables
        if self.categorical_trim is not None:
            for name, threshold in self.categorical_trim.items():
                value_counts = df[name].value_counts(normalize=True)
                drop_levels = set(value_counts[value_counts < threshold].index)
                remaining_levels = list(
                    set(df[name].cat.categories) - drop_levels)
                self.adjusted_categoricals[
                    name] = pandas.api.types.CategoricalDtype(remaining_levels)

        self.is_fit = True

    def transform(self, df):
        if not self.is_fit:
            raise Exception('Must fit first!')

        result_pieces = []

        # first, binarize features
        if len(self.binarized_fields) != 0:
            binarized = (df[self.binarized_fields]
                         .fillna(0)
                         .ne(0)
                         .astype('uint8'))
            binarized.rename(lambda name: 'nonzero_' + name, axis=1,
                             inplace=True)
            result_pieces.append(binarized)

        # next rescale / fill in NaNs for continuous variables
        scaled_continuous = df[self.continuous_cols_in_order].copy()
        scaled_continuous[:] = self.scaler.transform(scaled_continuous)
        if self.missing == 'sentinel':
            scaled_continuous.fillna(self.sentinel_value, inplace=True)
        elif self.missing == 'impute':
            # after standardizing, the means are 0,
            # so this represents mean imputation
            scaled_continuous.fillna(0, inplace=True)
        result_pieces.append(scaled_continuous)

        # lastly handle categorical variables
        categorical_df = df.select_dtypes(['category']).copy()
        if categorical_df.shape[1] > 0:
            for name, dtype in self.adjusted_categoricals.items():
                categorical_df[name] = categorical_df[name].astype(dtype)
            categorical_df = pd.get_dummies(categorical_df)
            result_pieces.append(categorical_df)

        # alphabetize and return
        res = pd.concat([piece.reindex(columns=sorted(piece.columns))
                         for piece in result_pieces],
                        axis=1)
        return res
