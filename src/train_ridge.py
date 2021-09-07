import argparse
import yaml
import pickle
import os
import pandas as pd
from sklearn import linear_model
from mlflow import log_param
from CustomPipeline import *


N_BINS = 16

def train_model(args, params):
    """
    train and save pickle file of model

    :args: input and output folders
    :params: params of model
    """
    # input = './data/prepare/prepared_train.csv'
    # output = './models'

    input = args.input
    output = args.output

    for k, v in params.items():
        log_param(k, v)

    train = pd.read_csv(input, index_col='id')

    X = train.query("target > 600").drop(["target"], axis=1)
    y = train.query("target > 600")["target"]

    lr = linear_model.Ridge(**params)
    model = LinearWrapper(lr, bins_linear=N_BINS)
    model.fit(X, y)

    # create dir if it isn't exists
    # if not os.path.exists(output):
    #     os.makedirs(output)
    # save
    with open(output, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open('params_model_linear.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    train_model(args, params)
    # train_model()



