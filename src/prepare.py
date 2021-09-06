import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepared():
    csv_path = '../data/train_anomaly.csv'
    directory = '../data/prepared'

    df = pd.read_csv(csv_path, index_col='id')
    df['target'] = 100 * df['target']
    train, val = train_test_split(df, test_size=0.1, random_state=42)

    # create dir if it isn't exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save to csv
    train.to_csv(directory + '/prepared_train.csv')
    val.to_csv(directory + '/prepared_val.csv')


if __name__ == '__main__':
    # parse arguments from console
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", required=True)
    # parser.add_argument("--output", required=True)
    # args = parser.parse_args()
    # train_model(args)
    train_model()
