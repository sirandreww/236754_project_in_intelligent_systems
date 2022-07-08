"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Process

from data_set import get_data_set

"""
***********************************************************************************************************************
    Test Bench Class
***********************************************************************************************************************
"""


class TestBench:
    """
    Class that takes some other class of a time series forecasting architecture, and tests it on
    multiple types of data.
    """

    def __init__(
            self,
            class_to_test,
            metrics_and_apps_to_test=[
                ("container_mem", "kube-rbac-proxy"),
                ("container_mem", "cni-plugins"),
                ("container_mem", "driver-registrar"),
                ("container_cpu", "kube-rbac-proxy"),
                ("container_cpu", "oauth-proxy"),
                ("container_cpu", "collector"),
                ("node_mem", "moc/smaug"),
                ("node_mem", "emea/balrog"),
            ],
            test_percentage=0.2,
            sub_sample_rate=5,
            path_to_data="../data/",
    ):
        self.__class_to_test = class_to_test
        self.__metrics_and_apps_to_test = metrics_and_apps_to_test
        self.__test_percentage = test_percentage
        self.__sub_sample_rate = sub_sample_rate
        self.__path_to_data = path_to_data

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    def __get_data(self, metric, app):
        dataset = get_data_set(
            metric=metric,
            application_name=app,
            path_to_data=self.__path_to_data
        )
        dataset.sub_sample_data(sub_sample_rate=self.__sub_sample_rate)
        dataset.normalize_data()
        return dataset.split_to_train_and_test(test_percentage=self.__test_percentage)

    @staticmethod
    def __plot_result(original, prediction_as_np_array):
        original_as_series = original["sample"].copy()
        predicted_as_series = pd.Series(prediction_as_np_array)
        x_axis = [time for time in original["time"]]
        original_as_series.index = x_axis
        predicted_as_series.index = x_axis[-len(prediction_as_np_array):]
        ax = original_as_series.plot()
        predicted_as_series.plot(ax=ax)
        plt.show()
        # plt.savefig('predict%d.pdf' % i)
        # plt.close()

    def __do_one_test(self, metric, app):
        print(f"Fetching data for metric='{metric}', app='{metric}'.")
        train, test = self.__get_data(
            metric=metric,
            app=app,
        )
        print("Making an instance of the class we want to test")
        model = self.__class_to_test()
        print("Starting training loop")
        model.learn_from_data_set(training_data_set=train)
        print("Starting testing loop")
        total_mse = 0
        for i, test_sample in enumerate(test):
            how_much_to_give = len(test_sample) // 2
            how_much_to_predict = len(test_sample) - how_much_to_give
            returned_ts_as_np_array = model.predict(
                ts_as_df_start=test_sample[: how_much_to_give],
                how_much_to_predict=how_much_to_predict
            )
            # make sure the output is in the right format
            assert isinstance(returned_ts_as_np_array, np.ndarray)
            assert len(returned_ts_as_np_array) == how_much_to_predict
            assert returned_ts_as_np_array.shape == (how_much_to_predict,)
            assert returned_ts_as_np_array.dtype == np.float64
            # plot only first 10 results
            if i < 10:
                self.__plot_result(
                    original=test_sample,
                    prediction_as_np_array=returned_ts_as_np_array,
                )
            out_should_be = test_sample["sample"].to_numpy()[how_much_to_give:]
            mse_here = (np.square(out_should_be - returned_ts_as_np_array)).mean()
            total_mse += mse_here
        print(f"Done with metric='{metric}', app='{metric}', the sum of the mse over the test batch is = {total_mse}.")

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def run_training_and_tests(self):
        print("Powering on test bench")
        for metric, app in self.__metrics_and_apps_to_test:
            print(f"testing metric='{metric}', app='{metric}'.")
            self.__do_one_test(
                metric=metric,
                app=app,
            )
        print("Powering off test bench")


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    tester = TestBench(
        class_to_test=DumbPredictor,
    )
    tester.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
