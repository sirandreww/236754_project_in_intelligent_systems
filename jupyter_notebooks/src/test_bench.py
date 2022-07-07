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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random

"""
***********************************************************************************************************************
    Data Merger Class
***********************************************************************************************************************
"""


class TestBench:
    """
    Class that takes some other class of a time series forecasting architecture, and tests it on
    multiple types of data.
    """

    def __init__(self, class_to_test, metrics_and_apps_to_test):
        self.__class_to_test = class_to_test
        self.__metrics_and_apps_to_test = metrics_and_apps_to_test

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

    def __getitem__(self, key):
        return self.__list_of_df[key]

    def __len__(self):
        return len(self.__list_of_df)

    def sub_sample_data(self, sub_sample_rate):
        new_list_of_df = []

        for df in self:
            sub_sampled_data = df.groupby(df.index // sub_sample_rate).max()
            assert len(sub_sampled_data) == ((len(df) + sub_sample_rate - 1) // sub_sample_rate)
            new_list_of_df.append(sub_sampled_data)

        self.__list_of_df = new_list_of_df

    def plot_dataset(self, number_of_samples):
        samples = random.sample(self.__list_of_df, k=number_of_samples)
        for df in samples:
            # plt.close("all")
            ts = df["sample"].copy()
            ts.index = [time for time in df["time"]]
            ts.plot()
            plt.show()

    def normalize_data(self):
        assert not self.__is_data_normalized
        self.__is_data_normalized = True
        min_sample, max_sample = self.__get_min_and_max_of_list_of_df()
        self.__max = max_sample
        self.__min = min_sample
        # print("max_sample = ", max_sample, " min_sample = ", min_sample)
        for df in self:
            normalized_sample_column = (df["sample"] - min_sample) / (max_sample - min_sample)
            # print(normalized_sample_column)
            df["sample"] = normalized_sample_column
            assert 0 <= df["sample"].min() and df["sample"].max() <= 1
        # check
        for df in self:
            assert 0 <= df["sample"].min() and df["sample"].max() <= 1


"""
***********************************************************************************************************************
    get train and test datasets
***********************************************************************************************************************
"""


def __get_names_of_json_files_in_directory(directory_path):
    csv_names = [f for f in listdir(directory_path) if (isfile(join(directory_path, f)) and ("json" in f))]
    return csv_names


def __get_names_of_relevant_files(metric, path_to_data):
    list_of_files = __get_names_of_json_files_in_directory(path_to_data)
    relevant_files = [file for file in list_of_files if (metric in file)]
    relevant_files.sort()
    return relevant_files


def __get_data_as_list_of_df_from_file(data_dict, application_name):
    result_list = []
    relevant_keys = [k for k in data_dict.keys() if (application_name in k)]
    for k in relevant_keys:
        list_of_ts = data_dict[k]
        for time_series in list_of_ts:
            start_time = datetime.strptime(time_series["start"], "%Y-%m-%d %H:%M:%S")
            stop_time = datetime.strptime(time_series["stop"], "%Y-%m-%d %H:%M:%S")
            date_time_range = [start_time + timedelta(minutes=i) for i in range(len(time_series["data"]))]
            assert date_time_range[-1] == stop_time
            time_series_as_df = pd.DataFrame(
                {
                    "sample": time_series["data"],
                    "time": date_time_range
                },
                # index=date_time_range
            )
            result_list.append(time_series_as_df)
    return result_list


def __get_data_as_list_of_df(metric, application_name, path_to_data):
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    result_list = []
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            result_list += __get_data_as_list_of_df_from_file(
                data_dict=data_dict,
                application_name=application_name
            )
    return result_list


def get_data_set(metric, application_name, path_to_data):
    # constants
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]

    # checks
    assert metric in __supported_metrics

    list_of_df = __get_data_as_list_of_df(
        metric=metric,
        application_name=application_name,
        path_to_data=path_to_data
    )

    ds = TimeSeriesDataSet(list_of_df=list_of_df)
    return ds


def get_amount_of_data_per_application(metric, path_to_data):
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]
    assert metric in __supported_metrics
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    application_names_histogram = {}
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            for k in data_dict.keys():
                app_name = k.split(", ")[0]
                # count number of time series samples
                amount_of_data = 0
                for ts in data_dict[k]:
                    amount_of_data += len(ts["data"])
                # add count to running count
                if app_name in application_names_histogram:
                    application_names_histogram[app_name] += amount_of_data
                else:
                    application_names_histogram[app_name] = amount_of_data
    result = sorted(application_names_histogram.items(), key=lambda item: - item[1])
    return result


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    print("Start.")
    test = 1
    if test == 0:
        print("Getting DataSet.")
        dataset = get_data_set(
            metric="container_mem",
            application_name="bridge-marker",
            path_to_data="../data/"
        )
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=10)
        print("Subsampling.")
        dataset.sub_sample_data(sub_sample_rate=5)
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=10)
        print("Normalizing.")
        dataset.normalize_data()
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=10)
    else:
        hist = get_amount_of_data_per_application(
            metric="container_mem",
            path_to_data="../data/"
        )
        print(hist)


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
