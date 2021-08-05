import pandas as pd

ID2NAME = [
    "readcount_all.csv",
    "normalized_count_with_deseq2.csv",
]


def read_feature(path):
    df = pd.read_csv(path, index_col=0)
    return df.T
