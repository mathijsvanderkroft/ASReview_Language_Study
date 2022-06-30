import numpy as np
import pandas as pd
import os

try:
    from gensim.utils import simple_preprocess
    from gensim.models.doc2vec import TaggedDocument
    from gensim.models.doc2vec import Doc2Vec as GenSimDoc2Vec
    from gensim import corpora
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

ENp = r"datasets\EN\Kwok_2020.csv"
DEp = r"datasets\EN_XX\Kwok_2020_DE.csv"
ESp = r"datasets\EN_XX\Kwok_2020_ES.csv"
TRp = r"datasets\EN_XX\Kwok_2020_TR.csv"
# ENp = r"datasets\EN\Appenzeller-Herzog_2020.csv"
# DEp = r"datasets\EN_XX\Appenzeller-Herzog_2020_DE.csv"
# ESp = r"datasets\EN_XX\Appenzeller-Herzog_2020_ES.csv"
# TRp = r"datasets\EN_XX\Appenzeller-Herzog_2020_TR.csv"
# ENp = r"datasets\EN\van_de_Schoot_2017.csv"
# DEp = r"datasets\EN_XX\van_de_Schoot_2017_DE.csv"
# ESp = r"datasets\EN_XX\van_de_Schoot_2017_ES.csv"
# TRp = r"datasets\EN_XX\van_de_Schoot_2017_TR.csv"

# open csv file:
EN = pd.read_csv(ENp)
DE_EN = pd.read_csv(DEp)
ES_EN = pd.read_csv(ESp)
TR_EN = pd.read_csv(TRp)

sets = [EN, DE_EN, ES_EN, TR_EN]
setnames = ['EN', 'DE_EN', 'ES_EN', 'TR_EN']
sparselen = []

for idx, s in enumerate(sets):
    # concatenate all strings in column "abstract" and "title" into one string:
    df = s.fillna('')
    corpus = df['abstract'] + ' ' + df['title']
    dict = corpora.Dictionary([simple_preprocess(doc) for doc in corpus])

    # number of words in dictionary that only occur in a single document (sparse):
    sparselen.append(len([key for key, val in dict.token2id.items() if dict.dfs[val] == 1]))
    print("total dictionary size {}: {} - sparse words: {}".format(
        setnames[idx],
        len(dict.token2id),
        sparselen[idx]))

for idx, set in enumerate(setnames[1:]):
    print("sparse ratio {}:{} = {}:{}".format(
        setnames[0],
        setnames[idx+1],
        1,
        round(sparselen[idx + 1] / sparselen[0], 2)))

