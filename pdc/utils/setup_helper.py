import os

import git
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pdc.dataset import feature
from pdc.dataset import sample_info


def setup_base_result(args, result):
    try:
        git_hexsha = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True).head.commit.hexsha
    except Exception:
        git_hexsha = 'Failed to get git'
    result['git_hexsha'] = git_hexsha


def setup_classification_datasets(args, result):
    read_count_name = feature.ID2NAME[0]
    read_count_path = os.path.join(args.feature_dir, read_count_name)
    read_count_df = feature.read_feature(read_count_path)

    result['feature_id'] = args.feature_id
    feature_file_name = feature.ID2NAME[args.feature_id]
    result['feature_name'] = feature_file_name
    feature_path = os.path.join(args.feature_dir, feature_file_name)
    print("feature_path:", feature_path)
    feature_df = feature.read_feature(feature_path)

    feature_columns = list(set(read_count_df.columns).intersection(set(feature_df.columns)))
    print("rc columns:", len(read_count_df.columns), 'feature columns', len(feature_df.columns))
    print("use feature_columns:", feature_columns)
    print("diff feature columns:", set(read_count_df.columns).symmetric_difference(set(feature_df.columns)))
    read_count_df = read_count_df.loc[:, feature_columns]
    feature_df = feature_df.loc[:, feature_columns]
    sample_info_df = sample_info.read_sample_info(args.background)

    dataset_df = pd.merge(feature_df, sample_info_df, left_index=True, right_index=True, how='inner')
    print('used samples', dataset_df.index)
    print("total sample size:", len(dataset_df))
    result['total sample size'] = len(dataset_df)
    train_index, test_index = train_test_split(dataset_df.index.values, test_size=0.3, random_state=args.hold_out_seed)
    train_dataset_df = dataset_df.loc[train_index, :]
    test_dataset_df = dataset_df.loc[test_index, :]

    info_columns = ['Sex', 'Age']
    train_dataset = {}
    train_dataset['read_count'] = read_count_df.loc[train_dataset_df.index, feature_df.columns].astype(np.float64)
    train_dataset['gene_expression'] = train_dataset_df[feature_df.columns].astype(np.float64)
    train_dataset['label'] = train_dataset_df[['Class']].astype(np.float64)
    train_dataset['info'] = train_dataset_df[info_columns].astype(np.float64)
    train_dataset['yahr'] = train_dataset_df[['Yahr']].astype(np.float64)

    print("train size:", len(train_dataset['label']))
    result['train size'] = len(train_dataset['label'])

    test_dataset = None
    if test_dataset_df is not None:
        test_dataset = {}
        test_dataset['read_count'] = read_count_df.loc[test_dataset_df.index, feature_df.columns].astype(np.float64)
        test_dataset['gene_expression'] = test_dataset_df[feature_df.columns].astype(np.float64)
        test_dataset['label'] = test_dataset_df[['Class']].astype(np.float64)
        test_dataset['info'] = test_dataset_df[info_columns].astype(np.float64)
        test_dataset['yahr'] = test_dataset_df[['Yahr']].astype(np.float64)

        print("test size:", len(test_dataset['label']))
        result['test size'] = len(test_dataset['label'])
    return train_dataset, test_dataset


def setup_yahr_datasets(args, result):
    read_count_name = feature.ID2NAME[0]
    read_count_path = os.path.join(args.feature_dir, read_count_name)
    read_count_df = feature.read_feature(read_count_path)

    result['feature_id'] = args.feature_id
    feature_file_name = feature.ID2NAME[args.feature_id]
    result['feature_name'] = feature_file_name
    feature_path = os.path.join(args.feature_dir, feature_file_name)
    print("feature_path:", feature_path)
    feature_df = feature.read_feature(feature_path)

    feature_columns = list(set(read_count_df.columns).intersection(set(feature_df.columns)))
    print("rc columns:", len(read_count_df.columns), 'feature columns', len(feature_df.columns))
    print("use feature_columns:", feature_columns)
    print("diff feature columns:", set(read_count_df.columns).symmetric_difference(set(feature_df.columns)))
    read_count_df = read_count_df.loc[:, feature_columns]
    feature_df = feature_df.loc[:, feature_columns]
    sample_info_df = sample_info.read_sample_info(args.background)

    dataset_df = pd.merge(feature_df, sample_info_df, left_index=True, right_index=True, how='inner')
    print('used samples', dataset_df.index)
    print("total sample size:", len(dataset_df))
    result['total sample size'] = len(dataset_df)

    train_index, test_index = train_test_split(dataset_df.index.values, test_size=0.3, random_state=args.hold_out_seed)
    train_dataset_df = dataset_df.loc[train_index, :]
    test_dataset_df = dataset_df.loc[test_index, :]

    info_columns = ['Sex', 'Age']
    train_dataset = {}
    train_dataset['read_count'] = read_count_df.loc[train_dataset_df.index, feature_df.columns].astype(np.float64)
    train_dataset['gene_expression'] = train_dataset_df[feature_df.columns].astype(np.float64)
    train_dataset['label'] = train_dataset_df[['Yahr']].astype(np.float64)
    train_dataset['info'] = train_dataset_df[info_columns].astype(np.float64)

    print("train size:", len(train_dataset['label']))
    result['train size'] = len(train_dataset['label'])

    test_dataset = None
    if test_dataset_df is not None:
        test_dataset = {}
        test_dataset['read_count'] = read_count_df.loc[test_dataset_df.index, feature_df.columns].astype(np.float64)
        test_dataset['gene_expression'] = test_dataset_df[feature_df.columns].astype(np.float64)
        test_dataset['label'] = test_dataset_df[['Yahr']].astype(np.float64)
        test_dataset['info'] = test_dataset_df[info_columns].astype(np.float64)

        print("test size:", len(test_dataset['label']))
        result['test size'] = len(test_dataset['label'])
    return train_dataset, test_dataset
