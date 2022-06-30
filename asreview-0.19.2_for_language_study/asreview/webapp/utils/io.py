# Copyright 2019-2020 The ASReview Authors. All Rights Reserved.
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

import json
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from asreview import __version__ as asreview_version
from asreview.config import LABEL_NA
from asreview.data import ASReviewData
from asreview.webapp.utils.paths import get_data_file_path
from asreview.webapp.utils.paths import get_labeled_path
from asreview.webapp.utils.paths import get_pool_path
from asreview.webapp.utils.paths import get_proba_path
from asreview.webapp.utils.paths import get_project_path


class CacheDataError(Exception):
    pass


def _get_cache_data_path(project_id):

    fp_data = get_data_file_path(project_id)

    return get_data_file_path(project_id) \
        .with_suffix(fp_data.suffix + ".pickle")


def _read_data_from_cache(project_id, version_check=True):

    fp_data_pickle = _get_cache_data_path(project_id)

    try:
        # get the pickle data
        with open(fp_data_pickle, 'rb') as f_pickle_read:
            data_obj, data_obj_version = pickle.load(f_pickle_read)

        # validate data object
        if not isinstance(data_obj.df, pd.DataFrame):
            raise ValueError()

        # drop cache files generated by older versions
        if (not version_check) or (asreview_version == data_obj_version):
            return data_obj

    except FileNotFoundError:
        # file not available
        pass
    except Exception as err:
        # problem loading pickle file or outdated
        # remove the pickle file
        logging.error(f"Error reading cache file: {err}")
        try:
            os.remove(fp_data_pickle)
        except FileNotFoundError:
            pass

    raise CacheDataError()


def _write_data_to_cache(project_id, data_obj):

    fp_data_pickle = _get_cache_data_path(project_id)

    logging.info("Store a copy of the data in a pickle file.")
    with open(fp_data_pickle, 'wb') as f_pickle:
        pickle.dump((data_obj, asreview_version), f_pickle)


def read_data(project_id, use_cache=True, save_cache=True):
    """Get ASReviewData object from file.

    Parameters
    ----------
    project_id: str, iterable
        The project identifier.
    use_cache: bool
        Use the pickle file if available.
    save_cache: bool
        Save the file to a pickle file if not available.

    Returns
    -------
    ASReviewData:
        The data object for internal use in ASReview.

    """

    # use cache file
    if use_cache:
        try:
            return _read_data_from_cache(project_id)
        except CacheDataError:
            pass

    # load from file
    fp_data = get_data_file_path(project_id)
    data_obj = ASReviewData.from_file(fp_data)

    # save a pickle version
    if save_cache:
        _write_data_to_cache(project_id, data_obj)

    return data_obj


def read_pool(project_id):
    pool_fp = get_pool_path(project_id)
    try:
        with open(pool_fp, "r") as f:
            pool = json.load(f)
        pool = [int(x) for x in pool]
    except FileNotFoundError:
        pool = None
    return pool


def write_pool(project_id, pool):
    pool_fp = get_pool_path(project_id)
    with open(pool_fp, "w") as f:
        json.dump(pool, f)


def read_proba_legacy(project_id):
    """Read a project <0.15 proba values"""

    # get the old json project file path
    proba_fp = Path(get_project_path(project_id), "proba.json")

    with open(proba_fp, "r") as f:

        # read the JSON file and make a list of the proba's
        proba = json.load(f)
        proba = [float(x) for x in proba]

    # make a dataframe that looks like the new structure
    as_data = read_data(project_id)
    proba = pd.DataFrame(
        {
            "proba": [float(x) for x in proba]
        },
        index=as_data.record_ids
    )
    proba.index.name = "record_id"
    return proba


def read_proba(project_id):

    proba_fp = get_proba_path(project_id)
    try:
        return pd.read_csv(proba_fp, index_col="record_id")
    except FileNotFoundError:

        # try to read the legacy file
        try:
            return read_proba_legacy(project_id)
        except FileNotFoundError:
            # no proba.csv or proba.json found.
            pass

    return None


def write_proba(project_id, proba):

    # get the proba file path location
    proba_fp = get_proba_path(project_id)

    # validate object
    if not isinstance(proba, pd.DataFrame):
        raise ValueError("Expect pandas.DataFrame with proba values.")

    if proba.index.name != "record_id":
        raise ValueError("Expect index with name 'record_id'.")

    # write the file to a csv file
    proba.to_csv(proba_fp)


def read_label_history(project_id, subset=None):
    """Get all the newly labeled papers from the file.

    Make sure to lock the "active" lock.
    """

    try:
        with open(get_labeled_path(project_id), "r") as fp:
            labeled = json.load(fp)

        if subset is None:
            labeled = [[int(idx), int(label)] for idx, label in labeled]
        elif subset in ["included", "relevant"]:
            labeled = [[int(idx), int(label)] for idx, label in labeled
                       if int(label) == 1]
        elif subset in ["excluded", "irrelevant"]:
            labeled = [[int(idx), int(label)] for idx, label in labeled
                       if int(label) == 0]
        else:
            raise ValueError(f"Subset value '{subset}' not found.")

    except FileNotFoundError:
        # file not found implies that there is no file written yet
        labeled = []

    return labeled


def write_label_history(project_id, label_history):
    label_fp = get_labeled_path(project_id)

    with open(label_fp, "w") as f:
        json.dump(label_history, f)


def read_current_labels(project_id, label_history=None):
    """Function to combine label history with prior labels.

    Function that combines the label info in the dataset and
    the label history in the project file.
    """
    # read the asreview data
    as_data = read_data(project_id)

    # use label history from project file
    if label_history is None:
        label_history = read_label_history(project_id)

    # get the labels in the import dataset
    labels = as_data.labels

    # make a list of NA labels if None
    if labels is None:
        labels = np.full(len(as_data), LABEL_NA, dtype=int)

    # update labels with label history
    label_idx = [idx for idx, incl in label_history]
    label_incl = [incl for idx, incl in label_history]

    # create a pandas series such that the index can be used
    labels_s = pd.Series(labels, index=as_data.df.index)
    labels_s.loc[label_idx] = label_incl

    return np.array(labels_s, dtype=int)