"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
import numpy as np


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


def find_best_hyper_parameters(config, train_model, training_data_set, length_to_predict, split_vertically):
    from pytorch_lightning.callbacks import EarlyStopping
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    from ray.tune.schedulers import ASHAScheduler

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
            "MAE": "val_MeanAbsoluteError",
            "MAPE": "val_MeanAbsolutePercentageError",
        },
        on="validation_end",
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["MAE", "MAPE", "training_iteration"],
    )

    # resources_per_trial = {"cpu": 8, "gpu": 1}
    resources_per_trial = {"cpu": 2}

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
