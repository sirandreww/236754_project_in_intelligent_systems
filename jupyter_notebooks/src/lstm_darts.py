"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
from darts.models import BlockRNNModel
import test_bench
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError

"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class DartsLSTMTester:
    def __init__(self, longest_length_to_predict, shortest_input):
        # Early stop callback
        my_stopper = EarlyStopping(
            monitor="train_MeanAbsolutePercentageError",  # "val_loss",
            patience=5,
            min_delta=0.05,
            mode='min',
        )
        pl_trainer_kwargs = {"callbacks": [my_stopper]}
        self.model = BlockRNNModel(
            model="LSTM",
            input_chunk_length=shortest_input,
            output_chunk_length=shortest_input,
            n_epochs=500,
            torch_metrics=MeanAbsolutePercentageError(),
            pl_trainer_kwargs=pl_trainer_kwargs
        )
        self.__msg = "[DartsLSTMTester]"
        # # print
        # print(self.__msg, f"model = {self.driver.model}")
        # print(self.__msg, f"learning_rate =", learning_rate)
        # print(self.__msg, f"optimizer =", self.driver.optimizer)
        # print(self.__msg, f"batch_size =", self.driver.batch_size)
        # print(self.__msg, f"padding = {self.driver.padding}")
        # print(self.__msg, f"num_epochs =", self.driver.num_epochs)

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    @staticmethod
    def __get_data_as_list_of_np_arrays(training_data_set):
        training_data_set_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        return training_data_set_as_list_of_np

    def learn_from_data_set(self, training_data_set):
        list_of_np_array = self.__get_data_as_list_of_np_arrays(
            training_data_set=training_data_set,
        )
        list_of_series = [
            darts.timeseries.TimeSeries.from_values(arr)
            for arr in list_of_np_array
        ]
        # t_size = int(len(list_of_series) * 0.9)
        # train = list_of_series[:t_size]
        # val = list_of_series[t_size:]
        self.model.fit(list_of_series)

    def predict(self, ts_as_df_start, how_much_to_predict):
        series = darts.timeseries.TimeSeries.from_dataframe(ts_as_df_start, time_col="time", value_cols="sample")
        res = self.model.predict(n=how_much_to_predict, series=series)
        assert len(res) == how_much_to_predict
        res_np_arr = res.pd_series().to_numpy()
        return res_np_arr


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    tb = test_bench.TestBench(
        class_to_test=DartsLSTMTester,
        path_to_data="../data/",
        tests_to_perform=[
            {"metric": "node_mem", "app": "moc/smaug", "test percentage": 0.2, "sub sample rate": 5,
             "data length limit": 20},
        ]
    )
    tb.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
