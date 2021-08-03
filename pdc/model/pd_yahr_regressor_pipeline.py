import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from pdc.model.sklearn_like_pipeline import SklearnLikePipeline
from pdc.preprocessing.rank_converter import RankConverter
from pdc.preprocessing.read_count_feature_selector import ReadCountFeatureSelector


class PdYahrRegressorPipeline(SklearnLikePipeline):

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
        if fitting:
            model_params = self.parameters.get("model_params", {})
            etr_params = model_params.get("etr", {})
            self._setup_model(x, y, model_class=ExtraTreesRegressor, model_params=etr_params)
        return self._predict_with_model(x, return_proba=return_proba)
