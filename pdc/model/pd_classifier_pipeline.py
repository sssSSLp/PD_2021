import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from pdc.model.pd_yahr_regressor_pipeline import PdYahrRegressorPipeline
from pdc.model.sklearn_like_pipeline import SklearnLikePipeline
from pdc.preprocessing.read_count_feature_selector import ReadCountFeatureSelector
from pdc.preprocessing.rank_converter import RankConverter


class PdClassifierPipeline(SklearnLikePipeline):

    def fit_and_predict(self, *args, fitting=False, return_proba=False, **kwargs):
        x_read_count = kwargs['x_read_count']
        x_gene_expression = kwargs['x_gene_expression']
        x_info = kwargs['x_info']
        if fitting:
            columns = []
            columns.extend(x_gene_expression.columns)
            columns.extend(x_info.columns)
            y = kwargs['label']
        else:
            y = None

        if fitting:
            self.feature_selector_with_read_count = \
                ReadCountFeatureSelector(self.parameters.get("feature_selector_with_read_count", {}))
            self.feature_selector_with_read_count.fit(x=x_read_count)
        selected_x_gene_expression = self.feature_selector_with_read_count.transform(x=x_gene_expression)

        if fitting:
            self.rank_converter = \
                RankConverter(self.parameters.get("rank_converter", {}))
            self.rank_converter.fit(x=selected_x_gene_expression, y=y)
        normalized_x_gene_expression = self.rank_converter.transform(x=selected_x_gene_expression)
        x = pd.merge(normalized_x_gene_expression, x_info, left_index=True, right_index=True, how='inner')

        using_yahr_regressor = self.parameters.get("using_yahr_regressor", True)
        if using_yahr_regressor:
            if fitting:
                yahr_regressor_params = self.parameters.get("yahr_regressor", {})
                self.yahr_regressor = PdYahrRegressorPipeline(parameters=yahr_regressor_params)
                self.yahr_regressor.fit(
                    x_read_count=kwargs['x_read_count'],
                    x_gene_expression=kwargs['x_gene_expression'],
                    x_info=kwargs['x_info'],
                    label=kwargs['yahr'],
                )
            yahr_pred = self.yahr_regressor.predict(
                x_read_count=kwargs['x_read_count'],
                x_gene_expression=kwargs['x_gene_expression'],
                x_info=kwargs['x_info'])
            x = pd.merge(x, yahr_pred, left_index=True, right_index=True, how='inner')

        if fitting:
            model_params = self.parameters.get("model_params", {})
            etc_params = model_params.get("etc", {})
            self._setup_model(x, y, model_class=ExtraTreesClassifier, model_params=etc_params)
        return self._predict_with_model(x, return_proba=return_proba)
