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

from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

from asreview.models.feature_extraction.base import BaseFeatureExtraction

class Tfidf_DE(BaseFeatureExtraction):
    """TF-IDF feature extraction technique.

    Use the standard TF-IDF (Term Frequency-Inverse Document Frequency) feature
    extraction technique from `SKLearn <https://scikit-learn.org/stable/modules/
    generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`__. Gives a
    sparse matrix as output. Works well in combination with
    :class:`asreview.models.classifiers.NaiveBayesClassifier` and other fast
    training models (given that the features vectors are relatively wide).

    Arguments
    ---------
    ngram_max: int
        Can use up to ngrams up to ngram_max. For example in the case of
        ngram_max=2, monograms and bigrams could be used.
    stop_words: str, list or None
        When set to language name, use stopwords from that language. available languages:
            from nltk.corpus import stopwords
            print(stopwords.fileids())
        When passed as list, use the list of stopwords.
        If set to None or 'none', do not use stop words.
    """
    name = "tfidf_DE"
    label = "TF-IDF_DE"

    def __init__(self, *args, ngram_max=1, stop_words="german", **kwargs):
        """Initialize tfidf class.
        """
        super(Tfidf_DE, self).__init__(*args, **kwargs)
        self.ngram_max = ngram_max

        # If stop_words is passed as a list of stopwords, do not overwrite it:
        if isinstance(stop_words, str) and stop_words.lower() != "none":
            self.stop_words = stopwords.words(stop_words)
        else:
            self.stop_words = stop_words

        # check if stop_words are None, string or list:
        sklearn_stop_words = None
        if stop_words is None:
            sklearn_stop_words = None
            print("No stop words are used")
        elif isinstance(stop_words, str):
            if stop_words.lower() == "none":
                sklearn_stop_words = None
                print("No stop words are used")
            else:
                sklearn_stop_words = self.stop_words
        else:
            sklearn_stop_words = self.stop_words

        self._model = TfidfVectorizer(ngram_range=(1, ngram_max),
                                      stop_words=sklearn_stop_words)

    def fit(self, texts):
        self._model.fit(texts)

    def transform(self, texts):
        X = self._model.transform(texts).tocsr()
        return X

    def full_hyper_space(self):
        from hyperopt import hp

        hyper_space, hyper_choices = super(Tfidf_DE, self).full_hyper_space()
        hyper_choices.update({
            "fex_stop_words": ["english", "none"]
        })
        hyper_space.update({
            "fex_ngram_max": hp.uniformint("fex_ngram_max", 1, 3),
            "fex_stop_words": hp.choice('fex_stop_words',
                                        hyper_choices["fex_stop_words"]),
        })
        return hyper_space, hyper_choices
