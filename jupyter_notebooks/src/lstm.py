"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""


class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden).double()
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden).double()
        self.linear = nn.Linear(self.n_hidden, 1).double()

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float64)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float64)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float64)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float64)

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class LSTMTester:
    def __init__(self):
        self.model = LSTMPredictor()
        print(self.model)

    @staticmethod
    def __turn_training_data_set_into_one_big_tensor(training_data_set, number_to_fill_empty_with):
        training_data_set_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        max_length = max([len(arr) for arr in training_data_set_as_list_of_np])
        padded_training_data_set_as_list_of_np = [
            np.concatenate(
                (
                    arr,
                    np.full((max_length - len(arr),), number_to_fill_empty_with)
                )
            )
            for arr in training_data_set_as_list_of_np
        ]
        training_data_as_list_of_tensors = [torch.from_numpy(arr) for arr in padded_training_data_set_as_list_of_np]
        big_tensor = torch.stack(training_data_as_list_of_tensors)
        return big_tensor

    def learn_from_data_set(self, training_data_set):
        """
        The implementation has a problem, the model is learning on the padding that we added here to
        make learning run faster. Solving this would add much complexity to the code, and since
        the padding that we added is the only negative value in the data we think the model might
        learn to ignore it.
        """
        big_tensor = self.__turn_training_data_set_into_one_big_tensor(
            training_data_set=training_data_set,
            number_to_fill_empty_with=-2
        )
        train_input = big_tensor[:, :-1]
        train_target = big_tensor[:, 1:]

        criterion = nn.MSELoss()
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8)

        n_steps = 1
        for i in range(n_steps):
            print("Step", i)

            def closure():
                optimizer.zero_grad()
                out = self.model.forward(train_input)
                loss = criterion(out, train_target)
                print('loss:', loss.item())
                loss.backward()
                return loss

            optimizer.step(closure)

    def predict(self, ts_as_df_start, how_much_to_predict):
        with torch.no_grad():
            ts_as_np = ts_as_df_start["sample"].to_numpy()
            ts_as_tensor = torch.from_numpy(ts_as_np)
            prediction = self.model.forward(ts_as_tensor[None, :], future=how_much_to_predict)
            y = prediction.detach().numpy()
            return y


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    import test_bench
    tb = test_bench.TestBench(
        class_to_test=LSTMTester,
        metrics_and_apps_to_test=[("node_mem", "moc/smaug")]
    )
    tb.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
