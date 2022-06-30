from asreview.analysis import Analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

create_csv = True
add_WSS = True
add_RRF = True
add_ATD = True

root = "output"
outdoc = 'metrics_analysis.csv'
repeats = 15
languages = ['EN', 'DE', 'ES', 'TR']
feature_extractors = ['tfidf', 'fasttext', 'doc2vec', 'multilingual_pre']
    # options: 'tfidf', 'fasttext', 'doc2vec', 'multilingual_post'
classifiers = ['logistic']
datasets = ["van_de_Schoot_2017"]
paths = []
names = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if any(set in name for set in datasets) and \
                any('\\' + l + '\\' in path for l in languages) and \
                any('\\' + e + '\\' in path for e in feature_extractors) and \
                any('\\' + c in path for c in classifiers):
            # assert len(files) % repeats == 0, "Nr of files is not multiple of expected repeats in " + path
            paths.append(path)
            names.append(name)

# create dataframe
df = pd.DataFrame(columns=['path', 'name', 'dataset', 'language', 'feature_extractor', 'classifier', 'repeat', 'WSS@95%', 'RRF@10%', 'ATD'])

# fill the path and name column
df.path = paths
df.name = names

# fill the dataset column
for row in df.itertuples():
    for dataset in datasets:
        if dataset in row.name:
            df.at[row.Index, 'dataset'] = dataset

# fill the languages column
language = []
for path in paths:
    for l in languages:
        if '\\' + l + '\\' in path:
            language.append(l)
df.language = language

# fill the feature_extractor column
feature_extractor = []
for path in paths:
    for e in feature_extractors:
        if e in path:
            feature_extractor.append(e)
df.feature_extractor = feature_extractor

# fill the classifier column
classifier = []
for path in paths:
    for c in classifiers:
        if c in path:
            classifier.append(c)
df.classifier = classifier

# fill the repeat column
if (len(df) / repeats).is_integer():
    repeat = list(range(1, repeats+1)) * int((len(df) / repeats))
else:
    raise ValueError('Number of files not divisible by repeats')
df.repeat = repeat

# check if df contains a NaN:
if df.iloc[:, :-3].isnull().values.any():
    raise ValueError('df contains NaN, please check')

print("nr of files: {}".format(len(df)))

# fill the metrics columns
if create_csv and (add_WSS or add_RRF or add_ATD):
    for index, row in df.iterrows():
        new_analysis = Analysis.from_path(os.path.join(row['path'], row['name']))
        if add_WSS:
            df.iloc[index, df.columns.get_loc('WSS@95%')] = \
                new_analysis.wss(val=95, x_format='percentage')[0]
        if add_RRF:
            df.iloc[index, df.columns.get_loc('RRF@10%')] = \
                new_analysis.rrf(val=10, x_format='percentage')[0]
        if add_ATD:
            df.iloc[index, df.columns.get_loc('ATD')] = \
                np.mean(list(new_analysis.avg_time_to_discovery(result_format="percentage").values()))
        print("Processing file nr: {}".format(index))

# save the dataframe to a csv
if create_csv:
    df.to_csv(outdoc, index=False)