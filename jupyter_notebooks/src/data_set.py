"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import pandas as pd
import random
from os import listdir
from os.path import isfile, join
import json
__data_path = "../data/"

"""
***********************************************************************************************************************
    Data Merger Class
***********************************************************************************************************************
"""


class TimeSeriesDataSet:
    """
    Class that houses time series data set.
    """
    def __init__(self):
        pass

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """
    pass


"""
***********************************************************************************************************************
    get train and test datasets
***********************************************************************************************************************
"""

def __get_names_of_json_files_in_directory(directory_path):
    csv_names = [f for f in listdir(directory_path) if (isfile(join(directory_path, f)) and ("json" in f))]
    return csv_names


def __get_names_of_relevant_files(metric):
    list_of_files = __get_names_of_json_files_in_directory(__data_path)
    relevant_files = [file for file in list_of_files if (metric in file)]
    relevant_files.sort()
    return relevant_files


def __get_data_as_list_of_df():
    file_names = __get_names_of_relevant_files(metric=metric)

    for file_name in file_names:
        with open(f'{__data_path}{file_name}') as json_file:
            data_dict = json.load(json_file)
            relevant_keys = [k for k in data_dict.keys() if (application_name in k)]
            for k in relevant_keys:
                list_of_ts = data_dict[k]


def get_data_set(metric, subsample_rate, application_name):
    # constants
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]

    # checks
    assert metric in __supported_metrics
    assert 1 <= subsample_rate <= 100


    ds = TimeSeriesDataSet()
    return ds


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    dataset = get_data_set(
        metric="container_mem",
        subsample_rate=3,
        application_name="bridge-marker"
    )

    "driver-registrar"


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
