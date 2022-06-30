# Copyright 2019-2022 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

try:
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import fasttext.util
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

from asreview.models.feature_extraction.base import BaseFeatureExtraction

def _check_libs():
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Install gensim package to use"
            " FastText.")
    if not FASTTEXT_AVAILABLE:
        raise ImportError(
            "Install fasttext package to use"
            " FastText.")

class FastText_EN(BaseFeatureExtraction):
    """FastText feature extraction technique.
    This technique calculates word vectors from a library, even for out of vocabulary words.
    However, the implementation for generating sentence vectors from word vectors is
    a relatively simple l2 normalized average of the word vectors."""

    name = "fasttext_EN"
    label = "FastText_EN"

    def __init__(self, *args, lang_id='en', **kwargs):
        """Initialize the fasttext model."""
        super(FastText_EN, self).__init__(*args, **kwargs)
        fasttext.util.download_model(lang_id, if_exists='ignore')
        self.lang_id = lang_id

    def transform(self, texts):

        # check if gensim and fasttext is available
        _check_libs()

        ft = fasttext.load_model("cc.%s.300.bin" % self.lang_id)
        corpus = [' '.join(simple_preprocess(text)) for text in texts]
        X = np.asarray([ft.get_sentence_vector(text) for text in corpus])
        return X


