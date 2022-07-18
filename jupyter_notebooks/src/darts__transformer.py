"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

from darts.models import TransformerModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError
import darts__helper as dh
import torch


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class DartsTransformerTester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        self.__length_of_shortest_time_series = length_of_shortest_time_series

        # constants
        self.__msg = "[DartsTransformerTester]"

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
        
        if torch.cuda.is_available():
            pl_trainer_kwargs = {"callbacks": [my_stopper], "accelerator": "gpu", "gpus": [0]}
        else:
            pl_trainer_kwargs = {"callbacks": [my_stopper]}

        # Create the model
        model = TransformerModel(
            # model specific
            input_chunk_length=length_of_shortest_time_series // 2,
            output_chunk_length=1,
            d_model=16,  # 64,
            nhead=8,  # 4,
            num_encoder_layers=2,  # 3,
            num_decoder_layers=2,  # 3,
            dim_feedforward=128,  # 512,
            dropout=0.1,
            activation="relu",
            # shared for all models
            loss_fn=torch.nn.L1Loss(),
            batch_size=32,
            n_epochs=100,
            # optimizer_kwargs={"lr": 0.001},
            torch_metrics=torch_metrics,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
        )
        return model

    def learn_from_data_set(self, training_data_set):
        # dh.find_best_hp_for_tcn(
        #     length_of_shortest_time_series=self.__length_of_shortest_time_series,
        #     training_data_set=training_data_set
        # )
        assert min(len(df) for df in training_data_set) >= self.__length_of_shortest_time_series
        list_of_series = [dh.get_darts_series_from_df(ts_as_df) for ts_as_df in training_data_set]
        self.__model = self.__make_model(
            length_of_shortest_time_series=self.__length_of_shortest_time_series
        )
        self.__model.fit(list_of_series)

    def predict(self, ts_as_df_start, how_much_to_predict):
        return dh.predict_using_model(
            model=self.__model,
            ts_as_df_start=ts_as_df_start,
            how_much_to_predict=how_much_to_predict,
            length_of_shortest_time_series=self.__length_of_shortest_time_series
        )
