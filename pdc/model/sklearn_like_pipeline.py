import numpy as np
import pandas as pd


class SklearnLikePipeline(object):

    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None
        self.y_columns = None
        self.is_fitted = False

    def fit_and_predict(self, *args, fitting=False, return_proba=False, **kwargs):
        pass

    def _setup_model(self, x, y, model_class, model_params):
        #print("model_class", model_class, model_params)
        self.model = model_class(**model_params)
        self.y_columns = y.columns
        x_values = x.values
        y_values = y.values
        self.model.fit(X=x_values, y=y_values)
        self._setup_feature_importances(x=x)
        return self.model

    def _predict_with_model(self, x, return_proba):
        x_index = x.index
        x = x.values
        if return_proba:
            y_pred = self.model.predict_proba(X=x)
            _, n_labels = y_pred.shape
            columns = []
            assert len(self.y_columns) == 1
            for label_i in range(n_labels):
                c = f'{self.y_columns[0]}_{label_i}'
                columns.append(c)
            return pd.DataFrame(y_pred, index=x_index, columns=columns)
        else:
            y_pred = self.model.predict(X=x)
            return pd.DataFrame(y_pred, index=x_index, columns=self.y_columns)

    def fit(self, *args, **kwargs):
        self.fit_and_predict(*args, **kwargs, fitting=True)
        self.is_fitted = True
        return self

    def predict(self, *args, **kwargs):
        if not self.is_fitted:
            RuntimeError('you should call fit() before predict()')
        return self.fit_and_predict(*args, **kwargs, fitting=False)

    def predict_proba(self, *args, **kwargs):
        if not self.is_fitted:
            RuntimeError('you should call fit() before predict()')
        return self.fit_and_predict(*args, **kwargs, fitting=False, return_proba=True)

    def _setup_feature_importances(self, x):
        self.feature_importances = pd.DataFrame({'feature_importance': np.zeros(len(x.columns))}, index=x.columns)
        if hasattr(self.model, 'feature_importances_'):
            model_feature_importances = self.model.feature_importances_
            for i, name in enumerate(x.columns):
                self.feature_importances.loc[name] = model_feature_importances[i]
