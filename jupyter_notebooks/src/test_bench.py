"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

from multiprocessing import Process
import data_set as ds

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

    def __init__(self, class_to_test, metrics_and_apps_to_test=None, test_percentage=0.2):
        self.__class_to_test = class_to_test
        if metrics_and_apps_to_test is None:
            self.__metrics_and_apps_to_test = [
                ("container_mem", "kube-rbac-proxy"),
                ("container_mem", "cni-plugins"),
                ("container_mem", "driver-registrar"),
                ("container_cpu", "kube-rbac-proxy"),
                ("container_cpu", "oauth-proxy"),
                ("container_cpu", "collector"),
                ("node_mem", "moc/smaug"),
                ("node_mem", "emea/balrog"),
            ]
        else:
            self.__metrics_and_apps_to_test = metrics_and_apps_to_test
        self.__test_percentage = test_percentage

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    @staticmethod
    def __get_data(metric, app, test_percentage):
        dataset = ds.get_data_set(
            metric=metric,
            application_name=app,
            path_to_data="../data/"
        )
        dataset.sub_sample_data(sub_sample_rate=5)
        dataset.normalize_data()
        return dataset.split_to_train_and_test(test_percentage=test_percentage)

    def __do_one_test(self, metric, app, test_percentage):
        train, test = self.__get_data(metric=metric, app=app, test_percentage=test_percentage)
        model = self.__class_to_test()
        for train_sample in train:
            model.learn_from_time_series(train_sample)
        total_mse = 0
        for test_sample in test:
            how_much_to_give = len(test_sample) // 2
            returned_ts = model.predict(test_sample[: how_much_to_give], len(test_sample) - how_much_to_give)
            mse = 0  # (returned_ts - test_sample[how_much_to_give:])**2
            total_mse += mse
        print("total mse = ", total_mse)

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def run_training_and_tests(self):
        for metric, app in self.__metrics_and_apps_to_test:
            self.__do_one_test(metric=metric, app=app, test_percentage=self.__test_percentage)


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    class DumbPredictor:
        def __init__(self):
            pass

        def learn_from_time_series(self, ts_as_df):
            pass

        def predict(self, ts_as_df_start, how_much_to_predict):
            return ts_as_df_start[: how_much_to_predict]

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
