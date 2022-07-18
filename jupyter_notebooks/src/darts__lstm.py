"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

from darts.models import RNNModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError
import darts__helper as dh
import torch


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class DartsLSTMTester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        self.__length_of_shortest_time_series = length_of_shortest_time_series

        # constants
        self.__msg = "[DartsLSTMTester]"

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
        pl_trainer_kwargs = {"callbacks": [my_stopper], }  # "accelerator": "gpu", "gpus": [0]}

        # Current best trial: ebf27_00003 with MAPE=0.21706010401248932 and parameters={'input_chunk_length': 8, 'output_chunk_length': 1, 'model': 'LSTM', 'hidden_dim': 167, 'n_rnn_layers': 6, 'dropout': 0.08185994482725292, 'training_length': 16, 'batch_size': 112}
        # 'input_chunk_length': 8, 'output_chunk_length': 1, 'model': 'LSTM', 'hidden_dim': 144, 'n_rnn_layers': 5, 'dropout': 0.10242930865517266, 'training_length': 16, 'batch_size': 16, 'loss_fn': MSELoss()


        # Create the model
        model = RNNModel(
            # model specific
            input_chunk_length=length_of_shortest_time_series // 2,
            output_chunk_length=1,
            model="LSTM",
            hidden_dim=144,
            n_rnn_layers=5,
            dropout=0.10242930865517266,
            training_length=length_of_shortest_time_series - 1,
            # shared for all models
            # loss_fn=torch.nn.L1Loss(),
            batch_size=16,
            n_epochs=100,
            # optimizer_kwargs={"lr": 0.001},
            torch_metrics=torch_metrics,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
        )
        return model

    def learn_from_data_set(self, training_data_set):
        # dh.find_best_hp_for_lstm(
        #     length_of_shortest_time_series=self.__length_of_shortest_time_series,
        #     training_data_set=training_data_set
        # )
        assert min(len(df) for df in training_data_set) >= self.__length_of_shortest_time_series
        list_of_series = [dh.get_darts_series_from_df(ts_as_df) for ts_as_df in training_data_set]
        self.__model = DartsLSTMTester.__make_model(
            length_of_shortest_time_series=self.__length_of_shortest_time_series
        )
        self.__model.fit(list_of_series)

    def predict(self, ts_as_df_start, how_much_to_predict):
        assert self.__model is not None
        assert len(ts_as_df_start) >= self.__length_of_shortest_time_series
        series = dh.get_darts_series_from_df(ts_as_df_start)
        res = self.__model.predict(n=how_much_to_predict, series=series, verbose=False)
        res_np_arr = dh.get_np_array_from_series(series=res)
        assert len(res_np_arr) == how_much_to_predict
        assert res_np_arr.shape == (how_much_to_predict,)
        return res_np_arr
