import numpy as np
import time

try:
    from sentence_transformers.SentenceTransformer import SentenceTransformer  # noqa
except ImportError:
    ST_AVAILABLE = False
else:
    ST_AVAILABLE = True

from asreview.models.feature_extraction.base import BaseFeatureExtraction


def _check_st():
    if not ST_AVAILABLE:
        raise ImportError(
            "Install sentence-transformers package"
            " to use distiluse-base-multilingual-cased-v2.")


class MultilingualSentenceTransformer_pre(BaseFeatureExtraction):
    """distiluse-base-multilingual-cased-v2 feature extraction technique.

    Feature extraction technique based on distiluse-base-multilingual-cased-v2.
    Implementation based on the `sentence_transformers
    <https://github.com/UKPLab/sentence- transformers>`__ package. It is
    relatively slow.

    .. note::

        This feature extraction technique requires ``sentence_transformers``
        to be installed. Use ``pip install sentence_transformers`` or install
        all optional ASReview dependencies with ``pip install asreview[all]``

    """

    name = "multilingual_pre"
    label = "Multilingual Sentence Transformer that saves the feature matrix"

    def transform(self, texts):

        _check_st()

        model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased-v2')
        X = np.array(model.encode(texts))
        np.save(r'C:\Users\mijnn\Desktop\Data_Science\Thesis\language_study\sbert_model' + time.strftime('%H%M%S'), X)
        return X
