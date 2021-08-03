def convert_to_rank(gene_expression_matrix, normalizing=False):
    ret = gene_expression_matrix.rank(axis=1)
    if normalizing:
        ret = ret / len(gene_expression_matrix.columns)
    return ret


class RankConverter(object):
    def __init__(self, parameters):
        self.normalizing = parameters.get("normalizing", False)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return convert_to_rank(x, self.normalizing)
