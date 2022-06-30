from asreview.analysis.analysis_fix import Analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metrics_analysis.csv')
figures_path = "figures"

leave_out = ['DE_EN', 'ES_EN', 'TR_EN']

# add dataset pseudonyms
df['pseudonym'] = np.nan
df.loc[df['dataset'] == 'ACEInhibitors', 'pseudonym'] = 'ACE'
df.loc[df['dataset'] == 'Kwok_2020', 'pseudonym'] = 'Virus'
df.loc[df['dataset'] == 'Hall_2012', 'pseudonym'] = 'Software'
df.loc[df['dataset'] == 'Appenzeller-Herzog_2020', 'pseudonym'] = 'Wilson'
df.loc[df['dataset'] == 'van_de_Schoot_2017', 'pseudonym'] = 'PTSD'
df.loc[df['dataset'] == 'Nagtegaal_2019', 'pseudonym'] = 'Nudging'
df.loc[df['feature_extractor'] == 'multilingual_post', 'feature_extractor'] = 'sbert'

# remove rows where the language is in the leave out list
df = df[~df['language'].isin(leave_out)]
df = df[~df['pseudonym'].isin(['ACE'])]

for dataset in df.dataset.unique():
    for e in df.feature_extractor.unique():
        for c in df.classifier.unique():
            df_subset = df[(df['dataset'] == dataset) & (df['feature_extractor'] == e) & (df['classifier'] == c)]
            for path in df_subset.path.unique():
                lan = df.loc[df['path'] == path]['language'].iloc[0]
                mean_WSS = round(df_subset.loc[df_subset['path'] == path]['WSS@95%'].mean(), 1)
                mean_RRF = round(df_subset.loc[df_subset['path'] == path]['RRF@10%'].mean(), 1)
                mean_ATD = round(df_subset.loc[df_subset['path'] == path]['ATD'].mean(), 1)
                label = "{0}; WSS: {1}%; RRF: {2}%, ATD: {3}%".format(lan, mean_WSS, mean_RRF, mean_ATD)

                new_analysis = Analysis.from_path(path, prefix = dataset)
                inc = new_analysis.inclusions_found()
                # pad inc[2] to the length of inc[1] with zeros, errorbar array is a lot shorter than x and y arrays
                errorbar = np.pad(inc[2], (0, len(inc[1]) - len(inc[2])), 'constant', constant_values=0)
                markers, caps, bars = plt.errorbar(inc[0] * 100, inc[1] * 100, yerr = errorbar * 100, \
                             label = label)

                [bar.set_alpha(0.2) for bar in bars]
                print("processing %s" % path)
            name = df_subset.loc[:]['pseudonym'].iloc[0]
            title = 'Recall curve ' + name + ' ' + e
            plt.title(title)
            plt.xlabel('Labeled records (%)')
            plt.ylabel('Relevant records found (%)')
            plt.legend(loc='lower right')
            plt.savefig(figures_path + '\\' + title + '.png', dpi=1000)
            plt.savefig(figures_path + '\\' + title + '.pdf')

            plt.show()