"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError
import numpy as np


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class DartsTCNTester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        self.__length_of_shortest_time_series = length_of_shortest_time_series

        # constants
        self.__msg = "[DartsTCNTester]"

        # will change
        self.__model = None

    @staticmethod
    def __make_model(length_of_shortest_time_series):
        # A TorchMetric or val_loss can be used as the monitor
        torch_metrics = MeanAbsolutePercentageError()

        # Early stop callback
        my_stopper = EarlyStopping(
            monitor="train_MeanAbsolutePercentageError",
            patience=5,
            min_delta=0.001,
            mode='min',
        )
        pl_trainer_kwargs = {"callbacks": [my_stopper]}

        # Create the model
        model = TCNModel(
            # model specific
            input_chunk_length=length_of_shortest_time_series // 2,
            output_chunk_length=1,
            kernel_size=3,
            num_filters=3,
            num_layers=None,
            dilation_base=2,
            weight_norm=False,
            dropout=0.2,
            # shared for all models
            batch_size=128,
            n_epochs=100,
            optimizer_kwargs={"lr": 0.001},
            torch_metrics=torch_metrics,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        return model

    @staticmethod
    def __get_darts_series_from_df(df):
        arr = df["sample"].to_numpy()
        res = darts.timeseries.TimeSeries.from_values(arr)
        return res

    def learn_from_data_set(self, training_data_set):
        assert min(len(df) for df in training_data_set) >= self.__length_of_shortest_time_series
        list_of_series = [self.__get_darts_series_from_df(ts_as_df) for ts_as_df in training_data_set]
        self.__model = DartsTCNTester.__make_model(
            length_of_shortest_time_series=self.__length_of_shortest_time_series
        )
        self.__model.fit(list_of_series)

    def predict(self, ts_as_df_start, how_much_to_predict):
        assert self.__model is not None
        assert len(ts_as_df_start) >= self.__length_of_shortest_time_series
        series = self.__get_darts_series_from_df(ts_as_df_start)
        res = self.__model.predict(n=how_much_to_predict, series=series, verbose=False)
        res_np_arr = res.pd_series().to_numpy()
        assert isinstance(res_np_arr, np.ndarray)
        assert len(res_np_arr) == how_much_to_predict
        assert res_np_arr.shape == (how_much_to_predict,)
        assert res_np_arr.dtype == np.float64
        return res_np_arr
