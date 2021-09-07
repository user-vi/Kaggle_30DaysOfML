import argparse
import json
import pickle
import os
from sklearn.metrics import mean_squared_error
from CustomPipeline import *


def predict(args):
    """
    eval model
    """
    # input_df = './data/prepare/prepared_val.csv'
    # input_model = './models/model_ridge.pkl'
    # output = './scores.json'

    input_df = args.input_data
    input_model = args.input_model
    output = args.output

    df = pd.read_csv(input_df, index_col='id')

    X = df.drop(["target"], axis=1)
    y = df["target"]

    with open(input_model, "rb") as fd:
        model = pickle.load(fd)

    pred = model.predict(X)
    rmse = mean_squared_error(y, pred)

    # create dir if it isn't exists
    # mode = 'a' if os.path.exists(output) else 'w'
    # save
    with open(output, "w") as fd:
        json.dump({"rmse": rmse}, fd, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict(args)
    # predict()
