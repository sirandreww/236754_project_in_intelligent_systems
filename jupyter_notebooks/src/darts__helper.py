"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
import numpy as np
from darts.models import RNNModel
from darts.models import TCNModel

"""
***********************************************************************************************************************
    helper functions
***********************************************************************************************************************
"""


def __find_best_hyper_parameters(config, train_model, training_data_set, length_to_predict, split_vertically):
    from pytorch_lightning.callbacks import EarlyStopping
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    from ray.tune.schedulers import ASHAScheduler
    import torch 

    if split_vertically:
        train, test = training_data_set.split_to_train_and_test(length_to_predict=length_to_predict)
        val = [get_darts_series_from_df(ts_as_df) for ts_as_df in test]
        train = [get_darts_series_from_df(ts_as_df) for ts_as_df in train]
    else:
        train_and_val = [get_darts_series_from_df(ts_as_df) for ts_as_df in training_data_set]
        train = train_and_val[:len(train_and_val) // 2]
        val = train_and_val[len(train_and_val) // 2:]

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
            "EV": "val_ExplainedVariance",
            "MAE": "val_MeanAbsoluteError",
            "MAPE": "val_MeanAbsolutePercentageError",
        },
        on="validation_end",
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["EV", "MAE", "MAPE", "training_iteration"],
    )
    
    if torch.cuda.is_available():
        resources_per_trial = {"cpu": 2, "gpu": 1}
    else:
        resources_per_trial = {"cpu": 2}

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
        metric="EV",  # any value in TuneReportCallback.
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_darts",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


def __get_train_model_function(model_name):
    assert model_name in ["LSTM", "TCN", "DeepAR"]
    class_of_model = None
    if model_name in ["LSTM", "DeepAR"]:
        class_of_model = RNNModel
    elif model_name == "TCN":
        class_of_model = TCNModel

    def __train_model(model_args, callbacks, train, val):
        from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError, ExplainedVariance
        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError(), ExplainedVariance()])
        # Create the model using model_args from Ray Tune
        model = class_of_model(
            n_epochs=100,
            torch_metrics=torch_metrics,
            pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
            **model_args)

        model.fit(
            series=train,
            val_series=val,
        )
    return __train_model


"""
***********************************************************************************************************************
    api functions
***********************************************************************************************************************
"""


def get_darts_series_from_df(df):
    arr = np.float32(df["sample"].to_numpy())
    res = darts.timeseries.TimeSeries.from_values(arr)
    return res


def get_np_array_from_series(series):
    res_np_arr = np.float64(series.pd_series().to_numpy())
    assert isinstance(res_np_arr, np.ndarray)
    assert res_np_arr.dtype == np.float64
    return res_np_arr


def predict_using_model(model, ts_as_df_start, how_much_to_predict, length_of_shortest_time_series):
    assert model is not None
    assert len(ts_as_df_start) >= length_of_shortest_time_series
    series = get_darts_series_from_df(ts_as_df_start)
    res = model.predict(n=how_much_to_predict, series=series, verbose=False)
    res_np_arr = get_np_array_from_series(series=res)
    assert len(res_np_arr) == how_much_to_predict
    assert res_np_arr.shape == (how_much_to_predict,)
    return res_np_arr


def find_best_hp_for_lstm(length_of_shortest_time_series, training_data_set):
    from ray import tune
    import torch
    __find_best_hyper_parameters(
        config={
            "input_chunk_length": tune.choice([length_of_shortest_time_series // 2]),
            "output_chunk_length": tune.choice([1]),
            "model": tune.choice(["LSTM"]),
            "hidden_dim": tune.choice([16 * i for i in range(1, 30)]),
            "n_rnn_layers": tune.choice([1, 2]),
            "dropout": tune.choice([0.01 * i for i in range(1, 21)]),
            "training_length": tune.choice([length_of_shortest_time_series - 1]),
            # shared for all models
            "optimizer_kwargs": tune.choice([{"lr": 0.1}, {"lr": 0.05}, {"lr": 0.01}, {"lr": 0.005}, {"lr": 0.001}]),
            "batch_size": tune.choice([16 * i for i in range(2, 11)]),
            "loss_fn": tune.choice([torch.nn.MSELoss()])
        },
        train_model=__get_train_model_function(model_name="LSTM"),
        training_data_set=training_data_set,
        length_to_predict=None,
        split_vertically=False
    )


def find_best_hp_for_tcn(length_of_shortest_time_series, training_data_set):
    from ray import tune
    import torch
    __find_best_hyper_parameters(
        config={
            "input_chunk_length": tune.choice([length_of_shortest_time_series // 2]),
            "output_chunk_length": tune.choice([1]),
            "kernel_size": tune.choice([1, 2, 3, 4, 5, 6, 7, 8]),
            "num_filters": tune.choice([i for i in range(1, 100)]),  
            "num_layers": tune.choice([i for i in range(1, 100)]),  
            "dilation_base": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "weight_norm": tune.choice([True, False]),
            "dropout": tune.uniform(0, 0.2),
            # shared for all models
            "loss_fn": tune.choice([torch.nn.L1Loss(), torch.nn.MSELoss()]),
            "batch_size": tune.choice([16 * i for i in range(1, 11)]),
        },
        train_model=__get_train_model_function(model_name="TCN"),
        training_data_set=training_data_set,
        length_to_predict=None,
        split_vertically=False
    )
    