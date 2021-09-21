"""There are modules for wrapping"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from catboost import Pool

def decorator_maker_with_arguments(bins):
    def my_decorator(func):
        def wrapper(self, X, y):
            num_train = X.select_dtypes([int, float])
            cat_train = X.select_dtypes(object)
            self.num = list(num_train)
            self.cat = list(cat_train)

            pipeline_num = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('normal', PowerTransformer()),
                ('scaling', StandardScaler()),
                ('bins', KBinsDiscretizer(n_bins=bins))
            ])
            pipeline_cat = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder(handle_unknown='ignore')),
            ])
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', pipeline_num, self.num),
                    ('cat', pipeline_cat, self.cat),
                ], remainder="drop")

            self.preprocessor.fit(X)
            X_transformed = self.preprocessor.transform(X)

            print(X_transformed.shape)
#             print(self.preprocessor)
#             print(X_transformed)
            return func(self, X_transformed, y)
        return wrapper
    return my_decorator


def my_decorator_predict(foo):
    def wrapper(self, X):
        X_transformed = self.preprocessor.transform(X)
        print(X_transformed.shape)
        return foo(self, X_transformed)
    return wrapper


class LinearWrapper(linear_model.Ridge):
    def __init__(self):
        super(LinearWrapper, self).__init__()
        self.preprocessor = None
        self.num = None
        self.cat = None

    @decorator_maker_with_arguments(3)
    def fit(self, X, y):
        super(linear_model.Ridge, self).fit(X, y)

    @my_decorator_predict
    def predict(self, X):
        return super(linear_model.Ridge, self).predict(X)

    def coef_(self):
        super(linear_model.Ridge, self).coef_


# model = LinearWrapper()
# model.fit(X, y)
# p = model.predict(X)
# print(p)