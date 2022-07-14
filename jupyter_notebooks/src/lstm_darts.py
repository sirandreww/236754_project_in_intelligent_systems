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
import os


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class DartsLSTMTester:
    def __init__(self, longest_length_to_predict, shortest_input, metric, app):
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
        self.shortest_input = shortest_input

    def learn_from_data_set(self, training_data_set):
        list_of_np_array = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        list_of_series = [
            darts.timeseries.TimeSeries.from_values(arr)
            for arr in list_of_np_array
        ]
        self.find_best_hyper_parameters(list_of_series=list_of_series)
        self.model.fit(list_of_series)
        # os.system("cls")

    def predict(self, ts_as_df_start, how_much_to_predict):
        series = darts.timeseries.TimeSeries.from_dataframe(ts_as_df_start, time_col="time", value_cols="sample")
        res = self.model.predict(n=how_much_to_predict, series=series)
        assert len(res) == how_much_to_predict
        res_np_arr = res.pd_series().to_numpy()
        return res_np_arr

    """
    ***********************************************************************************************************************
        Find best hyper parameters
    ***********************************************************************************************************************
    """

    def find_best_hyper_parameters(self, list_of_series):
        os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = str(1)
        from darts.models import NBEATSModel
        from darts.datasets import AirPassengersDataset
        from pytorch_lightning.callbacks import EarlyStopping
        import pandas as pd
        from darts.dataprocessing.transformers import Scaler
        from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError
        from ray import tune
        from ray.tune import CLIReporter
        from ray.tune.integration.pytorch_lightning import TuneReportCallback
        from ray.tune.schedulers import ASHAScheduler
        os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = str(1)

        def train_model(model_args, callbacks, train, val):
            torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
            # Create the model using model_args from Ray Tune
            model = NBEATSModel(
                input_chunk_length=24,
                output_chunk_length=12,
                n_epochs=500,
                torch_metrics=torch_metrics,
                pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
                **model_args)

            model.fit(
                series=train,
                val_series=val,
            )

        os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = str(1)

        random.shuffle(list_of_series)
        train_size = int(len(list_of_series) * 0.9)
        train = list_of_series[:train_size]
        val = list_of_series[train_size:]

        # Early stop callback
        my_stopper = EarlyStopping(
            monitor="val_MeanAbsolutePercentageError",
            patience=5,
            min_delta=0.05,
            mode='min',
        )

        # set up ray tune callback
        tune_callback = TuneReportCallback(
            {
                "loss": "val_Loss",
                "MAPE": "val_MeanAbsolutePercentageError",
            },
            on="validation_end",
        )

        # define the hyperparameter space
        config = {
            "batch_size": tune.choice([16, 32, 64, 128]),
            "num_blocks": tune.choice([1, 2, 3, 4, 5]),
            "num_stacks": tune.choice([32, 64, 128]),
            "dropout": tune.uniform(0, 0.2),
        }

        reporter = CLIReporter(
            parameter_columns=list(config.keys()),
            metric_columns=["loss", "MAPE", "training_iteration"],
        )

        # resources_per_trial = {"cpu": 8, "gpu": 1}
        resources_per_trial = {"cpu": 1}

        # the number of combinations to try
        num_samples = 10

        scheduler = ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2)

        train_fn_with_parameters = tune.with_parameters(
            train_model, callbacks=[my_stopper, tune_callback], train=train, val=val,
        )

        analysis = tune.run(
            train_fn_with_parameters,
            resources_per_trial=resources_per_trial,
            # Using a metric instead of loss allows for
            # comparison between different likelihood or loss functions.
            metric="MAPE",  # any value in TuneReportCallback.
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="tune_darts",
        )

        print("Best hyperparameters found were: ", analysis.best_config)
        # from pytorch_lightning.callbacks import EarlyStopping
        # import pandas as pd
        # from darts.dataprocessing.transformers import Scaler
        # from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError
        # from ray import tune
        # from ray.tune import CLIReporter
        # from ray.tune.integration.pytorch_lightning import TuneReportCallback
        # from ray.tune.schedulers import ASHAScheduler
        # import random
        #
        # def train_model(model_args, callbacks, train, val):
        #     torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
        #     # Create the model using model_args from Ray Tune
        #     model = BlockRNNModel(
        #         input_chunk_length=self.shortest_input,
        #         output_chunk_length=self.shortest_input,
        #         n_epochs=500,
        #         torch_metrics=torch_metrics,
        #         pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        #         **model_args)
        #
        #     model.fit(
        #         series=train,
        #         val_series=val,
        #     )
        #
        # random.shuffle(list_of_series)
        # train_size = int(len(list_of_series) * 0.9)
        # train = list_of_series[:train_size]
        # val = list_of_series[train_size:]
        #
        # # Early stop callback
        # my_stopper = EarlyStopping(
        #     monitor="val_MeanAbsolutePercentageError",
        #     patience=5,
        #     min_delta=0.05,
        #     mode='min',
        # )
        #
        # # set up ray tune callback
        # tune_callback = TuneReportCallback(
        #     {
        #         "loss": "val_Loss",
        #         "MAPE": "val_MeanAbsolutePercentageError",
        #     },
        #     on="validation_end",
        # )
        #
        # # define the hyperparameter space
        # config = {
        #     "batch_size": tune.choice([16, 32, 64, 128]),
        #     "num_blocks": tune.choice([1, 2, 3, 4, 5]),
        #     "num_stacks": tune.choice([32, 64, 128]),
        #     "dropout": tune.uniform(0, 0.2),
        #     "hidden_size": tune.choice([16, 32, 64, 128]),
        #     "n_rnn_layers": tune.choice([1, 2, 3, 4, 5]),
        #     "optimizer_kwargs": tune.choice([{"lr": 1e-1}, {"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}])
        # }
        #
        # reporter = CLIReporter(
        #     parameter_columns=list(config.keys()),
        #     metric_columns=["loss", "MAPE", "training_iteration"],
        # )
        #
        # # resources_per_trial = {"cpu": 8, "gpu": 1}
        # resources_per_trial = {"cpu": 6}
        #
        # # the number of combinations to try
        # num_samples = 10
        #
        # scheduler = ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2)
        #
        # train_fn_with_parameters = tune.with_parameters(
        #     train_model, callbacks=[my_stopper, tune_callback], train=train, val=val,
        # )
        #
        # analysis = tune.run(
        #     train_fn_with_parameters,
        #     resources_per_trial=resources_per_trial,
        #     # Using a metric instead of loss allows for
        #     # comparison between different likelihood or loss functions.
        #     metric="MAPE",  # any value in TuneReportCallback.
        #     mode="min",
        #     config=config,
        #     num_samples=num_samples,
        #     scheduler=scheduler,
        #     progress_reporter=reporter,
        #     name="tune_darts",
        # )
        #
        # print("Best hyperparameters found were: ", analysis.best_config)


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
