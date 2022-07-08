"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

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
    ):
        self.__class_to_test = class_to_test
        self.__metrics_and_apps_to_test = metrics_and_apps_to_test
        self.__test_percentage = test_percentage
        self.__sub_sample_rate = sub_sample_rate

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    def __get_data(self, metric, app):
        dataset = get_data_set(
            metric=metric,
            application_name=app,
            path_to_data="../data/"
        )
        dataset.sub_sample_data(sub_sample_rate=self.__sub_sample_rate)
        dataset.normalize_data()
        return dataset.split_to_train_and_test(test_percentage=self.__test_percentage)

    def __plot_result(self, ts_input_as_df, ts_output_as_df, prediction_as_np_array):
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        ts_input_as_np = ts_input_as_df["sample"].to_numpy()
        ts_output_as_np = ts_output_as_df["sample"].to_numpy()
        n = len(ts_input_as_np)
        future = len(ts_output_as_np)
        assert len(prediction_as_np_array) == future

        plt.plot(np.arange(n), ts_input_as_np[:n], 'b', linewidth=2.0)
        plt.plot(np.arange(n, n + future), ts_output_as_np[:n], 'b', linewidth=2.0)
        plt.plot(np.arange(n, n + future), yi[n:], 'r:', linewidth=2.0)
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
        for test_sample in test:
            how_much_to_give = len(test_sample) // 2
            how_much_to_predict = len(test_sample) - how_much_to_give
            returned_ts_as_np_array = model.predict(
                ts_as_df_start=test_sample[: how_much_to_give],
                how_much_to_predict=how_much_to_predict
            )
            assert isinstance(returned_ts_as_np_array, np.array)
            assert returned_ts_as_np_array.shape == how_much_to_predict
            self.__plot_result(
                ts_input_as_df=test_sample[: how_much_to_give],
                ts_output_as_df=test_sample[how_much_to_give:],
                prediction_as_np_array=returned_ts_as_np_array
            )
        print(f"Done with metric='{metric}', app='{metric}'.")

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
