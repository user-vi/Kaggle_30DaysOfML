import argparse
import os
import pandas as pd
import dvc
from sklearn.model_selection import train_test_split

def prepared(args):
    # input = './data/train_anomaly.csv'
    # output = './data/prepare'

    input = args.input
    output = args.output

    df = pd.read_csv(input, index_col='id')
    df['target'] = 100 * df['target']
    # X = df.drop('target', axis=1)
    # y = df['target']
    train, val = train_test_split(df, test_size=0.9, random_state=42)

    # create dir if it isn't exists
    if not os.path.exists(output):
        os.makedirs(output)
    # save to csv
    train.to_csv(output + '/prepared_train.csv', index=True, index_label='id')
    val.to_csv(output + '/prepared_val.csv', index=True, index_label='id')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepared(args)
    # prepared()
