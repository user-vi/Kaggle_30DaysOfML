import argparse
import yaml
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from CustomPipeline import *


N_BINS = 16

def train_model():
    """
    train and save pickle file of model
    """
    csv_path = '../data/prepared_train.csv'
    directory = '../'

    train = pd.read_csv(csv_path, index_col='id')

    X = train_estimators.query("target > 600").drop(["target"], axis=1)
    y = train_estimators.query("target > 600")["target"]

    lr = linear_model.Ridge()
    model = LinearWrapper(lr, bins_linear=N_BINS)
    model.fit(X, y)

    # create dir if it isn't exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save
    with open(directory + '/model_ridge.pkl', 'wb') as f:
        pickle.dump(model, f)



if __name__ == '__main__':
    # parse arguments from console
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", required=True)
    # parser.add_argument("--output", required=True)
    # args = parser.parse_args()
    # train_model(args)
    train_model()


