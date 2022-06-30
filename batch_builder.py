import os

batch = []
paths = ["EN", "EN_XX"]
    # options: "EN", "EN_XX", "EN_XX_EN"
inpaths = ["datasets/" + path for path in paths]
outpath = "output/"
outextension = ".h5"
datasets = ["Appenzeller-Herzog_2020", "van_de_Schoot_2017", "Nagtegaal_2019", "Kwok_2020", "Hall_2012", "ACEInhibitors"],

files = []
filepaths = []
orig_language = 'EN'
XX_languages = ['DE', 'ES', 'TR']
    # current options: 'DE', 'ES', 'TR'
feature_extractors = ["tfidf", "doc2vec", "multilingual_pre", "fasttext"]
    # options: "tfidf", "doc2vec", "multilingual_pre", "multilingual_post", "fasttext"
    # will use feature extractors tfidf_XX, multilingual_post_XX and fasttext_XX

    # Note: lot of processing time can be saved by running the feature extraction for sbert only once:
    #     - set repeat to 1
    #     - set feature_extractors to "multilingual_pre"
    #     - multilingual_pre will save its feature matrices in the root folder
    #     - build and run the batch
    #     - build a complete batch with multilingual_post and e.g. 15 repeats
    #               (and optionally other feature extractors)
    #     - previously created feature matrices are stored in sbert_feature_matrices/...
    # Remember that multilingual_post assumes the correct feature matrixes created by multilingual_pre are in the root folder

classifiers = ['logistic']
repeats = 15
queries = "min"

for path in inpaths:
    for file in os.listdir(path):
        if any(datasetname in file for datasetname in datasets):
            files.append(file)
            filepaths.append(path + "/" + file)

idx = 0
for n in range(len(files)):
    name = files[n].replace(".csv", "")
    for e in feature_extractors:
        for m in classifiers:
            for i in range(repeats):

                batch = batch + ["asreview simulate "]

                if name[-2:] == orig_language:
                    batch[idx] = batch[idx] + filepaths[n] + " --state_file " + outpath + name[-5:]
                elif name[-2:] in XX_languages:
                    batch[idx] = batch[idx] + filepaths[n] + " --state_file " + outpath + name[-2:]
                else:
                    batch[idx] = batch[idx] + filepaths[n] + " --state_file " + outpath + "EN"

                e_extension = ""
                if e == 'tfidf' or e == 'fasttext':
                    if name[-2:] in XX_languages: e_extension = "_" + name[-2:]
                    else: e_extension = "_" + orig_language
                if e == 'multilingual_post':
                    if name[-2:] in XX_languages: e_extension = "_" + name[-2:]
                    elif name[-2:] == orig_language: e_extension = "_" + name[-5:]
                    else: e_extension = "_" + orig_language

                batch[idx] = batch[idx] + "/" + e + "/" + m + "/" + name + "_" + str(1+i) + outextension + \
                                     " -e " + e + e_extension + " -m " + m + " --n_queries " + str(queries)

                idx = idx + 1

batch.insert(0, "set \"startstamp=%date% %time%\"")
batch.append("echo started: %startstamp%, finished: %date% %time%")

with open("simulation_batch.bat", "w") as f:
    for line in batch:
        f.write(line + "\n")




