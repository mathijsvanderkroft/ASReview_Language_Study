from pytest import mark
import numpy as np

from asreview.models.classifiers import get_classifier
from asreview.models.query import get_query_model
from asreview.models.query import list_query_strategies


@mark.parametrize("query_strategy", [
    "max",
    "random",
    "max_random",
    "uncertainty",
    "max_uncertainty",
    "cluster",
])
def test_query(query_strategy,
               n_features=50,
               n_sample=100,
               n_instances_list=[0, 1, 5, 50],
               n_train_idx=[0, 1, 5, 50]):
    classifier = get_classifier("rf")

    query_model = get_query_model(query_strategy)
    X = np.random.rand(n_sample, n_features)

    y = np.concatenate((np.zeros(n_sample // 2), np.ones(n_sample // 2)),
                       axis=0)
    print(X.shape, y.shape)
    order = np.random.permutation(n_sample)
    print(order.shape)
    X = X[order]
    y = y[order]
    sources = query_strategy.split('_')

    classifier.fit(X, y)

    assert isinstance(query_model.param, dict)
    assert query_model.name == query_strategy

    for n_instances in n_instances_list:
        for n_train in n_train_idx:
            shared = {"query_src": {}, "current_queries": {}}
            train_idx = np.random.choice(np.arange(n_sample),
                                         n_train,
                                         replace=False)
            pool_idx = np.delete(np.arange(n_sample), train_idx)
            query_idx, X_query = query_model.query(X, classifier, pool_idx,
                                                   n_instances, shared)
            check_integrity(query_idx, X_query, X, pool_idx, shared,
                            n_instances, sources)


def check_integrity(query_idx, X_query, X, pool_idx, shared, n_instances,
                    sources):
    # First check if the query_indices are valid
    assert len(query_idx) == n_instances
    assert len(query_idx) == len(np.unique(query_idx))
    for i, idx in enumerate(query_idx):
        assert idx in pool_idx
        assert np.all(X_query[i] == X[idx])
        assert idx in shared["current_queries"]

    for idx, src in shared["current_queries"].items():
        assert src in sources
        assert idx in query_idx


def test_query_general():
    assert len(list_query_strategies()) >= 4
