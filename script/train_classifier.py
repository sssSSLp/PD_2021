import argparse
import os
import pprint

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
import yaml

from pdc.model.pd_classifier_pipeline import PdClassifierPipeline
from pdc.utils.evaluate import evaluate_classifier
from pdc.utils.evaluate import evaluate_regressor
from pdc.utils import setup_helper
from pdc.utils.update_params_with_optuna import update_params_with_optuna


class Objective(object):

    def __init__(self, dataset, default_params, test_run):
        self.dataset = dataset
        self.default_params = default_params
        self.test_run = test_run

        self.n_splits = 10
        if self.test_run:
            self.trial = 2
        else:
            self.trial = 10
        self.alpha = 1.0

    def __call__(self, trial):
        # trial.suggest_int("feature_selector_with_read_count/min_read_count", 1, 100)
        # trial.suggest_categorical("feature_selector_with_read_count/ratio", [0.5, 0.8, 0.9, 0.95, 0.99])

        # trial.suggest_categorical("model_params/etc/n_estimators", [100, 150, 200, 300, 400, 500])
        # trial.suggest_loguniform("model_params/etc/max_depth", 2, 32)
        # trial.suggest_categorical("model_params/etc/min_samples_split", [4, 8, 16])
        # trial.suggest_categorical("model_params/etc/max_features", ['auto', 0.2, 0.4, 0.6, 0.8])

        optuna_params = trial.params
        params = update_params_with_optuna(self.default_params, optuna_params)

        score_list = []
        for _ in range(self.trial):
            y_pread_list = []
            y_gt_list = []
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=0)
            for i, (train_i, test_i) in enumerate(kfold.split(self.dataset['read_count'].values)):
                model = PdClassifierPipeline(parameters=params)
                model.fit(x_read_count=self.dataset['read_count'].iloc[train_i, :],
                          x_gene_expression=self.dataset['gene_expression'].iloc[train_i, :],
                          x_info=self.dataset['info'].iloc[train_i, :],
                          yahr=self.dataset['yahr'].iloc[train_i, :],
                          label=self.dataset['label'].iloc[train_i, :],)
                y_pred = model.predict_proba(x_read_count=self.dataset['read_count'].iloc[test_i, :],
                                             x_gene_expression=self.dataset['gene_expression'].iloc[test_i, :],
                                             x_info=self.dataset['info'].iloc[test_i, :])
                print(model.feature_importances.nlargest(20, 'feature_importance'))
                y_gt = self.dataset['label'].iloc[test_i, :]
                y_pread_list.append(y_pred)
                y_gt_list.append(y_gt)

            y_pread = pd.concat(y_pread_list)
            y_gt = pd.concat(y_gt_list)
            y_pread_values = y_pread.values
            y_gt_values = y_gt.loc[y_pread.index].values
            evaluation = evaluate_classifier(y_gt_values, y_pread_values)
            score = evaluation['roc_auc_1']
            score_list.append(score)
        print("scores", score_list, flush=True)
        print("mean, std", np.mean(score_list), np.std(score_list), flush=True)
        score = np.mean(score_list) - self.alpha * np.std(score_list)
        return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', metavar='PATH')
    parser.add_argument('--feature_id', type=int, default=1)
    parser.add_argument('--background', metavar='PATH')
    parser.add_argument('--params', metavar='PATH')
    parser.add_argument('--n_trials', '-n', type=int, default=100)
    parser.add_argument('--distributed_study_name', '-d', default=None)
    parser.add_argument('--out', default='result')
    parser.add_argument('--hold_out_seed', type=int, default=42)
    parser.add_argument('--test_run', action='store_true')
    args = parser.parse_args()

    if args.params is not None:
        with open(args.params) as f:
            default_params = yaml.load(f)
    else:
        default_params = {}

    result = {}
    setup_helper.setup_base_result(args, result)
    train_dataset, test_dataset = setup_helper.setup_classification_datasets(args, result)

    os.makedirs(args.out, exist_ok=True)
    if args.n_trials > 0:
        study_name = None
        storage = "sqlite:///{}/example.db".format(args.out)
        if args.distributed_study_name is not None:
            study_name = args.distributed_study_name
            storage = optuna.storages.RDBStorage(os.environ['OPTUNA_STORAGE'], {'pool_pre_ping': True})
        print("study_name", study_name)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0),
                                    study_name=study_name,
                                    storage=storage,
                                    direction='maximize',
                                    load_if_exists=True)
        objective = Objective(dataset=train_dataset,
                              default_params=default_params,
                              test_run=args.test_run)
        study.optimize(objective, n_trials=args.n_trials)
        best_optuna_value = study.best_value
        best_optuna_param = study.best_params
        print('\n', '-' * 5, 'RESULT', '-' * 5)
        print(f'BEST score: {best_optuna_value}')
        print('PARAMS:')
        pprint.pprint(best_optuna_param, indent=2)

        best_params = update_params_with_optuna(default_params, best_optuna_param)
        result['best score'] = best_optuna_value
        result['best params'] = best_params
    else:
        best_params = default_params

    y_pread_list = []
    y_gt_list = []
    importances = None
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_i, val_i) in enumerate(kfold.split(train_dataset['read_count'].values)):
        model = PdClassifierPipeline(parameters=best_params)
        model.fit(x_read_count=train_dataset['read_count'].iloc[train_i, :],
                  x_gene_expression=train_dataset['gene_expression'].iloc[train_i, :],
                  x_info=train_dataset['info'].iloc[train_i, :],
                  yahr=train_dataset['yahr'].iloc[train_i, :],
                  label=train_dataset['label'].iloc[train_i, :], )
        y_pred = model.predict_proba(x_read_count=train_dataset['read_count'].iloc[val_i, :],
                                     x_gene_expression=train_dataset['gene_expression'].iloc[val_i, :],
                                     x_info=train_dataset['info'].iloc[val_i, :])
        print(model.feature_importances.nlargest(20, 'feature_importance'))
        if i == 0:
            feature_importances = model.feature_importances.copy(deep=True)
            feature_importances.columns = [f'feature_importance_{i}']
            importances = feature_importances
        else:
            importances[f'feature_importance_{i}'] = model.feature_importances['feature_importance']
        y_gt = train_dataset['label'].iloc[val_i, :]
        y_pread_list.append(y_pred)
        y_gt_list.append(y_gt)

    y_pred = pd.concat(y_pread_list)
    y_gt = pd.concat(y_gt_list)
    y_pred_values = y_pred.values
    y_gt_values = y_gt.loc[y_pred.index].values
    evaluation = evaluate_classifier(y_gt_values, y_pred_values)

    print("val evaluation")
    pprint.pprint(evaluation, indent=2)
    result['val evaluation'] = evaluation

    y_pred_label_values = np.argmax(y_pred_values, axis=1)
    val_prediction_df = pd.DataFrame({'y_pred_label': y_pred_label_values,
                                      'y_gt': y_gt_values[:, 0],
                                      'y_pred_proba_0': y_pred_values[:, 0],
                                      'y_pred_proba_1': y_pred_values[:, 1]},
                                     index=y_gt.index)
    val_prediction_df.to_csv(os.path.join(args.out, 'val_prediction.csv'))
    importances.to_csv(os.path.join(args.out, 'val_feature_importances.csv'))

    if test_dataset is not None:
        model = PdClassifierPipeline(parameters=best_params)
        model.fit(x_read_count=train_dataset['read_count'],
                  x_gene_expression=train_dataset['gene_expression'],
                  x_info=train_dataset['info'],
                  yahr=train_dataset['yahr'],
                  label=train_dataset['label'],)
        y_pred = model.predict_proba(x_read_count=test_dataset['read_count'],
                                     x_gene_expression=test_dataset['gene_expression'],
                                     x_info=test_dataset['info'])
        print(model.feature_importances.nlargest(20, 'feature_importance'))
        model.feature_importances.to_csv(os.path.join(args.out, 'test_feature_importances.csv'))
        y_gt = test_dataset['label']
        y_pred_values = y_pred.values
        y_gt_values = y_gt.loc[y_pred.index].values
        evaluation = evaluate_classifier(y_gt_values, y_pred_values)
        print("test evaluation")
        pprint.pprint(evaluation, indent=2)
        result['test evaluation'] = evaluation

        y_pred_label_values = np.argmax(y_pred_values, axis=1)
        test_prediction_df = pd.DataFrame({'y_pred_label': y_pred_label_values,
                                           'y_gt': y_gt_values[:, 0],
                                           'y_pred_proba_0': y_pred_values[:, 0],
                                           'y_pred_proba_1': y_pred_values[:, 1]},
                                          index=y_gt.index)
        test_prediction_df.to_csv(os.path.join(args.out, 'test_prediction.csv'))
        joblib.dump(model, os.path.join(args.out, 'test_model.joblib'))

    with open(os.path.join(args.out, 'result.out'), 'w') as f:
        pprint.pprint(result, f)


if __name__ == '__main__':
    main()
