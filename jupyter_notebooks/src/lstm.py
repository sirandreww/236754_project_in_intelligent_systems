"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import pytorch_driver_for_test_bench
import torch.nn as nn
import torch.optim as optim
import test_bench

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
        self.model = nn.Sequential(
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
        self.output_size = output_size

    def forward(self, x, future):
        out = self.model(x)
        return out[:, :future]


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class LSTMTester:
    def __init__(self, longest_length_to_predict):
        # prepare parameters
        self.__msg = "[LSTMTester]"
        self.driver = pytorch_driver_for_test_bench.PytorchTester()
        self.driver.model = LSTMPredictor(
            input_size=1,
            output_size=longest_length_to_predict,
        ).to(self.driver.device)
        learning_rate = 0.001
        self.driver.optimizer = optim.Adam(self.driver.model.parameters(), lr=learning_rate)
        self.driver.batch_size = 64
        self.driver.padding = -99999
        self.driver.num_epochs = 20
        self.driver.sample_multiplier = 10  # Number of samples we will learn with is 1+2+3+ ... +sample_multiplier
        # To understand why look at its use
        # print
        print(self.__msg, f"model = {self.driver.model}")
        print(self.__msg, f"learning_rate =", learning_rate)
        print(self.__msg, f"optimizer =", self.driver.optimizer)
        print(self.__msg, f"batch_size =", self.driver.batch_size)
        print(self.__msg, f"padding = {self.driver.padding}")
        print(self.__msg, f"num_epochs =", self.driver.num_epochs)

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def learn_from_data_set(self, training_data_set):
        return self.driver.learn_from_data_set(training_data_set=training_data_set)

    def predict(self, ts_as_df_start, how_much_to_predict):
        return self.driver.predict(ts_as_df_start=ts_as_df_start, how_much_to_predict=how_much_to_predict)


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    tb = test_bench.TestBench(
        class_to_test=LSTMTester,
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
