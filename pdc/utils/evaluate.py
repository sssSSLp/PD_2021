import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def evaluate_classifier(y_gt, y_pred_proba):
    ret = {}
    ret['roc_auc_0'] = roc_auc_score(1 - y_gt, y_pred_proba[:, 0])
    ret['roc_auc_1'] = roc_auc_score(y_gt, y_pred_proba[:, 1])

    y_pred = np.argmax(y_pred_proba, axis=1)
    assert(y_pred.shape[0] == y_pred_proba.shape[0])
    ret['classification_report'] = classification_report(y_gt, y_pred, output_dict=True)
    ret['confusion_matrix'] = confusion_matrix(y_gt, y_pred)
    return ret


def evaluate_regressor(y_gt, y_pred):
    result = {}
    mse = metrics.mean_squared_error(y_gt, y_pred)
    result['mse'] = mse

    r2_score = metrics.r2_score(y_gt, y_pred)
    result['r2_score'] = r2_score

    correlation, pvalue = stats.spearmanr(y_gt, y_pred)
    result['spearman_r'] = correlation
    result['spearman_r_p_value'] = pvalue

    correlation, pvalue = stats.pearsonr(y_gt, y_pred)
    result['pearson_r'] = correlation
    result['pearson_r_p_value'] = pvalue

    return result
