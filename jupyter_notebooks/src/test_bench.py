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
import time

from data_set import get_data_set

"""
***********************************************************************************************************************
    plot_result
***********************************************************************************************************************
"""


def plot_result(original, prediction_as_np_array):
    original_as_series = original["sample"].copy()
    predicted_as_series = pd.Series(prediction_as_np_array)
    x_axis = [time for time in original["time"]]
    original_as_series.index = x_axis
    predicted_as_series.index = x_axis[-len(prediction_as_np_array):]
    ax = original_as_series.plot()
    predicted_as_series.plot(ax=ax)
    plt.show()


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
            path_to_data,
            tests_to_perform=[
                # node mem
                {"metric": "node_mem", "app": "moc/smaug", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                {"metric": "node_mem", "app": "emea/balrog", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                # container mem
                {"metric": "container_mem", "app": "nmstate-handler", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                {"metric": "container_mem", "app": "coredns", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                {"metric": "container_mem", "app": "keepalived", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                # container cpu
                {"metric": "container_cpu", "app": "kube-rbac-proxy", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                {"metric": "container_cpu", "app": "dns", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
                {"metric": "container_cpu", "app": "collector", "test percentage": 0.2, "sub sample rate": 5,
                 "data length limit": 20},
            ],
    ):
        self.__class_to_test = class_to_test
        self.__path_to_data = path_to_data
        for dictionary in tests_to_perform:
            assert "metric" in dictionary
            assert "app" in dictionary
            assert "test percentage" in dictionary
            assert "sub sample rate" in dictionary
            assert "data length limit" in dictionary
        self.__tests_to_perform = tests_to_perform
        self.__msg = "[TEST BENCH]"

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    def __get_data(self, dictionary):
        metric = dictionary["metric"]
        app = dictionary["app"]
        ss_rate = dictionary["sub sample rate"]
        dl_limit = dictionary["data length limit"]
        tp = dictionary["test percentage"]
        dataset = get_data_set(
            metric=metric,
            application_name=app,
            path_to_data=self.__path_to_data
        )
        print(self.__msg, f"Subsampling data from 1 sample per 1 minute to 1 sample per {ss_rate} minutes.")
        dataset.sub_sample_data(sub_sample_rate=ss_rate)
        print(self.__msg, f"Throwing out data that is less than {dl_limit * ss_rate} minutes long.")
        dataset.filter_data_that_is_too_short(data_length_limit=dl_limit)
        print(self.__msg, "Scaling data.")
        dataset.scale_data()
        print(self.__msg, "Splitting data into train and test")
        train, test = dataset.split_to_train_and_test(test_percentage=tp)
        print(self.__msg, f"Amount of train data is {len(train)}")
        print(self.__msg, f"Amount of test data is {len(test)}")
        return train, test

    @staticmethod
    def __get_mse_precision_recall_and_f1(original_np, predicted_np):
        assert len(original_np) == len(predicted_np)
        mse_here = (np.square(original_np - predicted_np)).mean()
        actual_positives = [original_np[i + 1] >= original_np[i] for i in range(len(original_np) - 1)]
        predicted_positives = [predicted_np[i + 1] >= predicted_np[i] for i in range(len(predicted_np) - 1)]
        assert len(actual_positives) == len(predicted_positives)
        true_positive = sum([
            1 if (og == predicted and predicted) else 0
            for og, predicted in zip(actual_positives, predicted_positives)
        ])
        false_positive = sum([
            1 if (og != predicted and predicted) else 0
            for og, predicted in zip(actual_positives, predicted_positives)
        ])
        false_negative = sum([
            1 if (og != predicted and not predicted) else 0
            for og, predicted in zip(actual_positives, predicted_positives)
        ])
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive != 0) else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative != 0) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall != 0) else 0
        return mse_here, precision, recall, f1

    def __give_one_test_to_model(self, test_sample, model, should_print):
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
        if should_print:
            plot_result(
                original=test_sample,
                prediction_as_np_array=returned_ts_as_np_array,
            )
        out_should_be = test_sample["sample"].to_numpy()[how_much_to_give:]
        mse_here, precision, recall, f1 = self.__get_mse_precision_recall_and_f1(
            original_np=out_should_be, predicted_np=returned_ts_as_np_array
        )
        return mse_here, precision, recall, f1

    def __print_report(self, metric, app, mse, precision, recall, f1, training_time):
        print(self.__msg, f"***********************************************************************")
        print(self.__msg, f"REPORT for                              metric='{metric}', app='{app}':")
        print(self.__msg, f"Training time in seconds is             {training_time}")
        print(self.__msg, f"Average mse over the test set is        {mse}")
        print(self.__msg, f"Average precision over the test set is  {precision}")
        print(self.__msg, f"Average recall over the test set is     {recall}")
        print(self.__msg, f"Average F1 over the test set is         {f1}")
        print(self.__msg, f"***********************************************************************")

    def __test_model_and_print_report(self, test, model, metric, app, training_time):
        total_mse = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        for i, test_sample in enumerate(test):
            mse_here, precision, recall, f1 = self.__give_one_test_to_model(
                test_sample=test_sample, model=model, should_print=(i < 10)
            )
            total_mse += mse_here
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        mse, precision, recall, f1 = total_mse / len(test), total_precision / len(test), total_recall / len(
            test), total_f1 / len(test)
        self.__print_report(metric=metric, app=app, mse=mse, precision=precision, recall=recall, f1=f1, training_time=training_time)
        print(self.__msg, f"Done with metric='{metric}', app='{app}'")
        return mse, precision, recall, f1, training_time

    def __do_one_test(self, dictionary):
        metric, app = dictionary["metric"], dictionary["app"]
        print(self.__msg, f"Fetching data for metric='{metric}', app='{app}'.")
        train, test = self.__get_data(dictionary=dictionary)
        print(self.__msg, "Making an instance of the class we want to test")
        model = self.__class_to_test()
        print(self.__msg, "Starting training loop")
        training_start_time = time.time()
        model.learn_from_data_set(training_data_set=train)
        training_stop_time = time.time()
        print(self.__msg, f"Training took {training_stop_time - training_start_time} seconds.")
        print(self.__msg, "Starting testing loop")
        return self.__test_model_and_print_report(test=test, model=model, metric=metric, app=app, training_time=training_stop_time-training_start_time)

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def run_training_and_tests(self):
        print(self.__msg, "Powering on test bench")
        full_report = []
        for dictionary in self.__tests_to_perform:
            app = dictionary["app"]
            metric = dictionary["metric"]
            print(self.__msg, f"testing metric='{metric}', app='{app}'.")
            mse, precision, recall, f1, training_time = self.__do_one_test(dictionary=dictionary)
            full_report += [(mse, precision, recall, f1, training_time)]
        assert len(full_report) == len(self.__tests_to_perform)
        for dictionary, (mse, precision, recall, f1, training_time) in zip(self.__tests_to_perform, full_report):
            app, metric = dictionary["app"], dictionary["metric"]
            self.__print_report(metric=metric, app=app, mse=mse, precision=precision, recall=recall, f1=f1, training_time=training_time)
        print(self.__msg, "Powering off test bench")


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    class DumbPredictor:
        def __init__(self):
            print("Constructor called.")

        def learn_from_data_set(self, training_data_set):
            print("Training started.")
            print("Training ending.")

        def predict(self, ts_as_df_start, how_much_to_predict):
            ts_as_np = ts_as_df_start["sample"].to_numpy()
            res = np.resize(ts_as_np, how_much_to_predict)
            return res

    tester = TestBench(
        class_to_test=DumbPredictor,
        path_to_data="../data/"
    )
    tester.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
