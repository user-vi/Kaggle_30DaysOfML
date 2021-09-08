import argparse
import os
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

def prepared(args):
    # input = './data/train_anomaly.csv'
    # output = './data/prepare'

    train = args.train
    test = args.test
    output = args.output

    train = pd.read_csv(train, index_col='id')
    test = pd.read_csv(test, index_col='id')

    test['label'] = 'test'
    train['label'] = 'train'

    df = pd.concat([test, train.drop('target', axis=1), test])

    num_train = df.select_dtypes([int, float])
    num = list(num_train)

    for i, m in enumerate(num):
        for j, n in enumerate(tqdm(num)):
            if (m != n) and (j > i):
                try:
                    kmeans = KMeans(n_clusters=9, random_state=42).fit(df[[m, n]])
                    df['k_means_' + m + n] = kmeans.labels_
                    df['k_means_' + m + n] = df['k_means_' + m + n].astype('object')
                except:
                    print(m, n, 'problem')

    test_with_clusters = df[df.label == 'test'].copy()
    train_with_clusters = df[df.label == 'train'].copy()
    train_with_clusters['target'] = train['target'].copy()


    # create dir if it isn't exists
    if not os.path.exists(output):
        os.makedirs(output)
    # save to csv
    train_with_clusters.to_csv(output + '/train.csv', index=True, index_label='id')
    test_with_clusters.to_csv(output + '/test.csv', index=True, index_label='id')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs='+')
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepared(args)
    # prepared()
