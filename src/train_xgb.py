import argparse
import yaml
import pickle
import xgboost
from CustomPipeline import *


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

    train = pd.read_csv(input, index_col='id')

    X = train.drop(["target"], axis=1)
    y = train["target"]

    xgb = xgboost.XGBRegressor(**params)
    model = XGBWrapper(xgb)
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

    with open('params_model_xgb.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    train_model(args, params)
    # train_model()



