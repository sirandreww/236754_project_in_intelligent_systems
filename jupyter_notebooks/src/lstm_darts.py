"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
from darts.models import RNNModel
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
    def __init__(self, length_to_predict, metric, app):
        # save inputs
        self.__metric = metric
        self.__app = app
        self.__length_to_predict = length_to_predict

        # constants
        self.__msg = "[DartsLSTMTester]"

        # will change
        self.__model = None

    def __get_hyper_parameters(self):
        # Hyper-Parameters found using 'find_best_hyper_parameters'
        hp_dict = {
            # node mem
            ('node_mem', 'moc/smaug'):   {'batch_size': 23, 'dropout': 0.1530952773311739, 'hidden_dim': 115, 'n_rnn_layers': 1, 'optimizer_kwargs': {'lr': 0.05}, 'output_chunk_length': 12}
            # ('node_mem', 'emea/balrog'): {'batch_size': 22, 'dropout': 0.09292093736642433, 'hidden_dim': 87, 'n_rnn_layers': 1, 'optimizer_kwargs': {'lr': 0.05}, 'output_chunk_length': 5},
            # container mem
            # ("container_mem", "nmstate-handler"): ,
            # ("container_mem", "coredns"): ,
            # ("container_mem", "keepalived"): ,
            # container cpu
            # ("container_cpu", "kube-rbac-proxy"): ,
            # ("container_cpu", "dns"): ,
            # ("container_cpu", "collector"): ,
        }
        hyper_parameters = hp_dict[(self.__metric, self.__app)] if ((self.__metric, self.__app) in hp_dict) else {
            'batch_size': 179,
            'dropout': 0.025308023037584018,
            'hidden_dim': 100,
            'n_rnn_layers': 3,
            'optimizer_kwargs': {'lr': 0.005},
            'output_chunk_length': self.__length_to_predict,
        }
        self.is_hyper_parameter_search_required = not ((self.__metric, self.__app) in hp_dict)
        if not self.is_hyper_parameter_search_required:
            print(self.__msg, f"Hyper-Parameters loaded for metric='{self.__metric}' and app='{self.__app}'")
        assert "batch_size" in hyper_parameters
        assert "dropout" in hyper_parameters
        assert "hidden_dim" in hyper_parameters
        assert "n_rnn_layers" in hyper_parameters
        assert "optimizer_kwargs" in hyper_parameters
        assert "output_chunk_length" in hyper_parameters
        return hyper_parameters

    def __make_model(self, list_of_series):
        hp = self.__get_hyper_parameters()
        self.input_chunk_length = min(len(df) - self.__length_to_predict for df in list_of_series)
        # A TorchMetric or val_loss can be used as the monitor
        torch_metrics = MeanAbsolutePercentageError()

        # Early stop callback
        my_stopper = EarlyStopping(
            monitor="train_MeanAbsolutePercentageError",  # "val_loss",
            patience=5,
            min_delta=0.05,
            mode='min',
        )
        pl_trainer_kwargs = {"callbacks": [my_stopper]}

        # Create the model
        self.__model = RNNModel(
            model="LSTM",
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=hp["output_chunk_length"],
            n_epochs=500,
            torch_metrics=torch_metrics,
            pl_trainer_kwargs=pl_trainer_kwargs,
            batch_size=hp["batch_size"],
            dropout=hp["dropout"],
            hidden_dim=hp["hidden_dim"],
            n_rnn_layers=hp["n_rnn_layers"],
            optimizer_kwargs=hp["optimizer_kwargs"]
        )

    def learn_from_data_set(self, training_data_set):
        list_of_np_array = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        list_of_series = [
            darts.timeseries.TimeSeries.from_values(arr)
            for arr in list_of_np_array
        ]
        self.__make_model(list_of_series)
        if self.is_hyper_parameter_search_required:
            self.find_best_hyper_parameters(list_of_series=list_of_series)
        self.__model.fit(list_of_series)

    def predict(self, ts_as_df_start, how_much_to_predict):
        series = darts.timeseries.TimeSeries.from_dataframe(ts_as_df_start, time_col="time", value_cols="sample")
        res = self.__model.predict(n=how_much_to_predict, series=series)
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
        from pytorch_lightning.callbacks import EarlyStopping
        from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError
        from ray import tune
        from ray.tune import CLIReporter
        from ray.tune.integration.pytorch_lightning import TuneReportCallback
        from ray.tune.schedulers import ASHAScheduler
        import random
        os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = str(1)

        def train_model(model_args, callbacks, train, val):
            torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
            # Create the model using model_args from Ray Tune
            model = RNNModel(
                model="LSTM",
                n_epochs=500,
                torch_metrics=torch_metrics,
                input_chunk_length=self.input_chunk_length,
                pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
                **model_args)

            model.fit(
                series=train,
                val_series=val,
            )

        os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = str(1)

        random.shuffle(list_of_series)
        train_size = int(len(list_of_series) * 0.8)
        train = list_of_series[:train_size]
        print(self.__msg, "len(train) =", len(train))
        val = list_of_series[train_size:]
        print(self.__msg, "len(val) =", len(val))

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
                "MAE": "val_MeanAbsoluteError",
                "MAPE": "val_MeanAbsolutePercentageError",
            },
            on="validation_end",
        )

        # define the hyperparameter space
        config = {
            "batch_size": tune.choice([i for i in range(0, 200)]),
            "dropout": tune.uniform(0, 0.2),
            "hidden_dim": tune.choice([i for i in range(1, 200)]),
            "n_rnn_layers": tune.choice([1, 2, 3, 4]),
            "optimizer_kwargs": tune.choice([
                {"lr": 0.1}, {"lr": 0.05}, {"lr": 0.01}, {"lr": 0.005}, {"lr": 0.001}, {"lr": 0.0005}, {"lr": 0.0001}
            ]),
            "output_chunk_length": tune.choice([i for i in range(1, self.__length_to_predict + 1)]),
        }

        reporter = CLIReporter(
            parameter_columns=list(config.keys()),
            metric_columns=["MAE", "MAPE", "training_iteration"],
        )

        # resources_per_trial = {"cpu": 8, "gpu": 1}
        resources_per_trial = {"cpu": 1}

        # the number of combinations to try
        num_samples = 100

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
