import argparse

import pandas as pd
import yaml
import pickle
from sklearn import linear_model
from CustomPipeline import *


def train_model(args):
    """
    train and save pickle file of model

    :args: input and output folders
    :params: params of model
    """
    # input = './data/prepare/prepared_train.csv'
    # output = './models'

    models = args.models
    val = args.input
    output = args.output

    val = pd.read_csv(val, index_col='id')
    X = val.drop('target', axis=1)

    X_estimator = pd.DataFrame()

    for i, path in enumerate(models):
        with open(path, "rb") as fd:
            model = pickle.load(fd)
            pred = model.predict(X)
            X_estimator[str(i)] = pred


    final_model = linear_model.Ridge()
    final_model.fit(X_estimator, val['target'])

    # create dir if it isn't exists
    # if not os.path.exists(output):
    #     os.makedirs(output)
    # save
    with open(output, 'wb') as f:
        pickle.dump(final_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, nargs='+')
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_model(args)
    # train_model()



