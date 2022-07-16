"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import pytorch__driver_for_test_bench
import torch.nn as nn

"""
***********************************************************************************************************************
    ExtractTensorAfterLSTM class
***********************************************************************************************************************
"""


# LSTM() returns tuple of (tensor, (recurrent state))
class ExtractTensorAfterLSTM(nn.Module):
    """
    Helper class that allows LSTM to be used inside nn.Sequential.
    Usually would be out right after nn.LSTM and right before nn.Linear.
    """

    @staticmethod
    def forward(x):
        # Output shape (batch, features, hidden)
        out, (h_n, c_t) = x
        # assert torch.equal(h_n, out)
        # Reshape shape (batch, hidden)
        # return tensor[:, -1, :]
        return out[:, -1, :]


"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMPredictor, self).__init__()
        hidden_size_for_lstm = 200
        hidden_size_for_linear = 32
        num_layers = 2
        dropout = 0.1
        self.__seq_model = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_for_lstm,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            ),
            ExtractTensorAfterLSTM(),
            # nn.ReLU(),
            nn.Linear(
                in_features=hidden_size_for_lstm,
                out_features=hidden_size_for_linear
            ),
            # nn.ReLU(),
            nn.Linear(
                in_features=hidden_size_for_linear,
                out_features=output_size
            )
        )

    def forward(self, x):
        out = self.__seq_model(x)
        return out


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class PytorchLSTMTester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        # prepare parameters
        self.__msg = "[PytorchLSTMTester]"
        self.__model_input_length = length_of_shortest_time_series // 2
        self.__model = LSTMPredictor(
            input_size=1,
            output_size=1,
        ).to(pytorch__driver_for_test_bench.get_device())
        # print
        print(self.__msg, f"model = {self.__model}")

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def learn_from_data_set(self, training_data_set):
        pytorch__driver_for_test_bench.train_neural_network(
            training_data_set=training_data_set,
            model=self.__model,
            model_input_length=self.__model_input_length
        )
        return None

    def predict(self, ts_as_df_start, how_much_to_predict):
        return self.driver.predict(ts_as_df_start=ts_as_df_start, how_much_to_predict=how_much_to_predict)


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    import framework__test_bench
    tb = framework__test_bench.TestBench(
        class_to_test=PytorchLSTMTester,
        path_to_data="../data/",
    )
    tb.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
