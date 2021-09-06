import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer
from catboost import Pool


class CustomPipeline:
    def __init__(self, n_bins: int, num_columns: list, cat_columns: list):
        self.n_bins = n_bins
        self.num = num_columns
        self.cat = cat_columns

    def get_preprocessor(self):
        pipeline_num = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaling', StandardScaler()),
            ('normal', PowerTransformer()),
            ('bins', KBinsDiscretizer(n_bins=self.n_bins))
        ])

        pipeline_cat = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoding', OneHotEncoder(handle_unknown='ignore')),
        ])

        return ColumnTransformer(
            transformers=[
                ('num', pipeline_num, self.num),
                ('cat', pipeline_cat, self.cat),
            ], remainder="drop")


class CustomModelWrapper:
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame):
        pass


class TypesOfColumns:
    def __init__(self, X: pd.DataFrame):
        self.X = X

        num_train = self.X.select_dtypes([int, float])
        cat_train = self.X.select_dtypes(object)
        self.num = list(num_train)
        self.cat = list(cat_train)
        self.idx = [X.columns.get_loc(i) for i in self.cat]

    def get_num(self) -> list:
        return self.num

    def get_cat(self) -> list:
        return self.cat

    def get_cat_idx(self) -> list:
        return self.idx


class LinearWrapper(CustomModelWrapper):

    def __init__(self, model, bins_linear: int):
        self.model = model
        self.bins = bins_linear
        self.num = None
        self.cat = None
        self.pipeline = None

    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num_train = X.select_dtypes([int, float])
        cat_train = X.select_dtypes(object)
        self.num = list(num_train)
        self.cat = list(cat_train)

        pipeline_num = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaling', StandardScaler()),
            ('normal', PowerTransformer()),
            ('bins', KBinsDiscretizer(n_bins=self.bins))
        ])
        pipeline_cat = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoding', OneHotEncoder(handle_unknown='ignore')),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', pipeline_num, self.num),
                ('cat', pipeline_cat, self.cat),
            ], remainder="drop")
        return preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series):
        preprocessor = self.get_preprocessor(X)

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('model', self.model),
                                        ])
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.array:
        return self.pipeline.predict(X)


class XGBWrapper(CustomModelWrapper):

    def __init__(self, model):
        self.model = model
        self.num = None
        self.cat = None
        self.pipeline = None

    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num_train = X.select_dtypes([int, float])
        cat_train = X.select_dtypes(object)
        self.num = list(num_train)
        self.cat = list(cat_train)

        pipeline_num = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaling', StandardScaler()),
            ('normal', PowerTransformer()),
        ])
        pipeline_cat = Pipeline(steps=[
            ('encoding', OneHotEncoder(handle_unknown='ignore')),
            #             ('encoding', OrdinalEncoder()),
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', pipeline_num, self.num),
                ('cat', pipeline_cat, self.cat),
            ], remainder="drop")
        return preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series):
        preprocessor = self.get_preprocessor(X)
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('model', self.model),
                                        ])
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.array:
        return self.pipeline.predict(X)


class CatBoostWrapper(CustomModelWrapper):

    def __init__(self, model):
        self.model = model
        self.cat_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cat_train = X.select_dtypes(object)
        cat = list(cat_train)
        self.cat_features = [X.columns.get_loc(i) for i in cat]

        train_pool = Pool(X, y, cat_features=self.cat_features)
        self.model.fit(train_pool, logging_level="Silent")

    def predict(self, X: pd.DataFrame) -> np.array:
        test_pool = Pool(X, cat_features=self.cat_features)
        return self.model.predict(test_pool)
