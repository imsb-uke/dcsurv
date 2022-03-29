import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class DataFrameTransformer(TransformerMixin):
    '''Returns transform as dataframe

    with specified suffix for transformed columns.
    '''

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns
        self._restore_category = False

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        elif isinstance(self.columns, str) and (self.columns == 'category'):
            self.columns = X.select_dtypes('category').columns
            self._restore_category = True

            self._original_categories = {}
            for col in X.select_dtypes('category').columns:
                self._original_categories[col] = X[col].cat.categories

        elif isinstance(self.columns, str) and (self.columns == 'not category'):
            self.columns = [col for col in X.columns if col not in X.select_dtypes('category')]
        elif all([col in X.columns for col in self.columns]):
            self.columns = self.columns
        else:
            raise ValueError('Specify valid columns')

        if len(self.columns) > 0:
            self.transformer.fit(X[self.columns], y)
        return self

    def transform(self, X):
        result = X.copy()

        if len(self.columns) == 0:
            return result

        result[self.columns] = self.transformer.transform(X[self.columns])

        if self._restore_category:
            result[self.columns] = result[self.columns].astype('category')

            for col in self.columns:
                result[col] = result[col].cat.set_categories(
                    self._original_categories[col])

        return result


class CatOneHotEncoder(TransformerMixin):

    '''Transforms all category columns to one hot that have at least 3 distinct values
    '''

    def __init__(self, **get_dummies_kwargs):
        self.get_dummies_kwargs = get_dummies_kwargs

    def fit(self, X, y=None):
        self.columns = X.select_dtypes('category').columns
        return self

    def transform(self, X):
        columns_to_oh = self.columns
        for col in self.columns:
            if len(X[col].value_counts()) <= 2:
                columns_to_oh = columns_to_oh.drop(col)

        return pd.get_dummies(X,
                              columns=columns_to_oh,
                              dtype='float32',
                              **self.get_dummies_kwargs)


class ColumnDropper:
    def __init__(self, drop):
        self.drop = drop

    def fit(self, X, y=None):
        self.columns = [col for col in X.columns
                        if col not in self.drop]
        return self

    def transform(self, X):
        return X[self.columns]


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_, y=None):
        return self

    def transform(self, input_, y=None):
        return input_


def get_pipeline_support():
    zero_impute_cols = [
        'mean_arterial_blood_pressure',
        'heart_rate',
        'respiration_rate',
        'serum_sodium']

    pipeline = get_pipeline_general()

    zero_imputer = DataFrameTransformer(
        SimpleImputer(missing_values=0, strategy='median'), columns=zero_impute_cols)

    pipeline.steps.insert(0, ('NumZeroImputer', zero_imputer))
    return pipeline


def get_pipeline_general():
    return Pipeline([
        ("StandardScaler", DataFrameTransformer(
            StandardScaler(), columns='not category')),
        ("NumImputer", DataFrameTransformer(
            SimpleImputer(strategy='median'), columns='not category')),
        ("CatImputer", DataFrameTransformer(
            SimpleImputer(strategy='most_frequent'), columns='category')),
        ("CatOneHotEncoder", CatOneHotEncoder(drop_first=True)),
        ("FloatConverter", FunctionTransformer(lambda x: x.astype('float32'))),
    ])


def get_pipeline_metabric():
    return get_pipeline_general()


def get_pipeline_flchain():
    return get_pipeline_general()


def get_pipeline(dataset_name):
    return {
        'support': get_pipeline_support(),
        'metabric': get_pipeline_metabric(),
        'flchain': get_pipeline_flchain(),
    }[dataset_name]
