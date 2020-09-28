import pathlib
import tempfile

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

DATASETS_DIR = pathlib.Path(__file__).resolve().parents[1]


def get_df_from_sklearn(subset):
    # setup
    categories = ['alt.atheism', 'soc.religion.christian']

    # download the data via sklearn and store in a temporary directory
    # that automatically gets cleaned from disk
    with tempfile.TemporaryDirectory() as tmp_dir:
        sklearn_dataset = fetch_20newsgroups(subset=subset,
                                             categories=categories,
                                             data_home=tmp_dir)

    # make a dataframe from the data
    df = pd.DataFrame({'article': sklearn_dataset.data,
                       'is_christian': sklearn_dataset.target})
    df['is_christian'] = df['is_christian'].astype('bool')
    return df


if __name__ == '__main__':
    # get the data
    print('Downloading data...')
    train_df = get_df_from_sklearn('train')
    test_df = get_df_from_sklearn('test')

    # find/make the output directory
    out_path = DATASETS_DIR / '20news'
    out_path.mkdir(exist_ok=True)

    print('Writing data to disk...')
    train_df.to_csv(out_path / 'train.csv', index=False)
    test_df.to_csv(out_path / 'test.csv', index=False)

    print('Done!')
