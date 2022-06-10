"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import pandas as pd

"""
***********************************************************************************************************************
    Data Merger Class
***********************************************************************************************************************
"""


class TimeSeriesDataSet:
    """
    Class that can merge fetched data from prometheus in operate first.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.list_of_rows_that_are_not_date_time = None
        self.df = self.__make_data_frame()

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    @staticmethod
    def __is_not_time_stamp(some_string: str):
        has_digit = any(char.isdigit() for char in some_string)
        has_line = some_string.count("-") > 0
        return not has_digit and not has_line

    def __get_new_name_of_column_i(self, df: pd.DataFrame, i: int) -> str:
        result_str = ""
        for j in range(len(self.list_of_rows_that_are_not_date_time) - 1):
            result_str = result_str + df.iloc[j][i] + " "
        result_str = result_str + df.iloc[len(self.list_of_rows_that_are_not_date_time) - 1][i]
        return result_str

    def __make_data_frame(self):
        # read dataframe from csv
        df = pd.read_csv(
            filepath_or_buffer=self.csv_path,
            index_col=0,
        )

        "rename columns (they are called 0, 1, 2 ... .They should be the name of the container/pod)"
        rows = df.index
        self.list_of_rows_that_are_not_date_time = list(filter(self.__is_not_time_stamp, list(rows)))
        old_names_to_new_names = {str(i): self.__get_new_name_of_column_i(df=df, i=i) for i in df.columns}
        df.rename(columns=old_names_to_new_names, inplace=True)

        "drop rows that makeup name of sample"
        df = df.iloc[len(self.list_of_rows_that_are_not_date_time):, :]

        "turn date time columns into datetime format"
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d_%H_%M_%S')  # %Y-04-19_17_00_00'

        "drop samples that have more than half of their data empty"
        df.dropna(thresh=(len(rows) / 2), axis=1, inplace=True)

        "drop rows that are entirely empty"  # dangerous
        df.dropna(how='all', inplace=True)

        # "drop rows that don't have enough unique values (easy to predict)"
        "cast to numeric"
        df = df.astype(dtype='float64')
        # df = df.to_numeric(, downcast='float')
        # print(df.dtypes)

        return df

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def get_data_frame(self):
        return self.df

    def get_dataset_description(self):
        return self.df.describe().transpose().to_markdown()

    def __str__(self):
        return self.df.to_markdown()


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    print("""
    This is only a library to make it easier to import data.
    This script should only be run for testing the library.
    The library takes a name of a merged file (merged with data_merger.py), and returns a dataset.
    """)

    ts_ds_container_cpu = TimeSeriesDataSet(
        "../data/container_cpu_usage_seconds_2022-04-19_17_00_00_to_2022-04-24_20_00_00__123_hours.csv")
    print(ts_ds_container_cpu)

    ts_ds_container_memory = TimeSeriesDataSet(
        "../data/container_memory_working_set_bytes_2022-04-19_17_00_00_to_2022-04-24_20_00_00__123_hours.csv")
    print(ts_ds_container_memory)

    ts_ds_node_memory = TimeSeriesDataSet(
        "../data/node_memory_active_bytes_percentage_2022-06-01_09_00_00_to_2022-06-06_10_00_00__121_hours.csv")
    print(ts_ds_node_memory)

    print("Done")


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
