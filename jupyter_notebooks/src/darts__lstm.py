"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
from darts.models import RNNModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError
import numpy as np
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
        pl_trainer_kwargs = {"callbacks": [my_stopper], "accelerator": "gpu", "gpus": [0]}

        # Best hyperparameters found were:  {'input_chunk_length': 8, 'output_chunk_length': 1, 'model': 'LSTM', 'hidden_dim': 60, 'n_rnn_layers': 1, 'dropout': 0.11171057976191785, 'training_length': 15, 'batch_size': 19}
        # Current best trial: ebf27_00003 with MAPE=0.21706010401248932 and parameters={'input_chunk_length': 8, 'output_chunk_length': 1, 'model': 'LSTM', 'hidden_dim': 167, 'n_rnn_layers': 6, 'dropout': 0.08185994482725292, 'training_length': 16, 'batch_size': 112}

        # Create the model
        model = RNNModel(
            # model specific
            input_chunk_length=length_of_shortest_time_series // 2,
            output_chunk_length=1,
            model="LSTM",
            hidden_dim=60,
            n_rnn_layers=1,
            dropout=0.11171057976191785,
            training_length=length_of_shortest_time_series - 1,
            # shared for all models
            loss_fn=darts.utils.losses.MAELoss(),
            batch_size=19,
            n_epochs=100,
            # optimizer_kwargs={"lr": 0.001},
            torch_metrics=torch_metrics,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
        )
        return model

    @staticmethod
    def train_model_for_hp_tuning(model_args, callbacks, train, val):
        from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError
        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
        # Create the model using model_args from Ray Tune
        model = RNNModel(
            n_epochs=100,
            torch_metrics=torch_metrics,
            pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
            **model_args)

        model.fit(
            series=train,
            val_series=val,
        )

    def learn_from_data_set(self, training_data_set):
        look_for_hp = True
        if look_for_hp:
            from ray import tune
            dh.find_best_hyper_parameters(
                config={
                    "input_chunk_length": tune.choice([self.__length_of_shortest_time_series // 2]),
                    "output_chunk_length": tune.choice([1]),
                    "model": tune.choice(["LSTM"]),
                    "hidden_dim": tune.choice([i for i in range(32, 200)]),
                    "n_rnn_layers": tune.choice([1, 2, 3, 4, 5, 6, 7]),
                    "dropout": tune.uniform(0, 0.2),
                    "training_length": tune.choice([self.__length_of_shortest_time_series - 1]),
                    # shared for all models
                    "batch_size": tune.choice([i for i in range(32, 200)]),
                    "loss_fn": tune.choice([darts.utils.losses.MAELoss(), torch.nn.MSELoss(), darts.utils.losses.MapeLoss()])
                },
                train_model=self.train_model_for_hp_tuning,
                training_data_set=training_data_set,
                length_to_predict=None,
                split_vertically=False
            )
        else:
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
