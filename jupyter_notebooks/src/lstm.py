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

    def __turn_training_data_set_into_one_big_tensor(self, training_data_set):
        training_data_set_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        training_data_set_as_list_of_tensors = [torch.from_numpy(arr) for arr in training_data_set_as_list_of_np]

    def learn_from_data_set(self, training_data_set):
        data_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        train_input_as_list_of_np = [arr[:-1] for arr in data_as_list_of_np]
        train_target_as_list_of_np = [arr[1:] for arr in data_as_list_of_np]
        train_input_list = [torch.from_numpy(arr) for arr in train_input_as_list_of_np]
        train_target_list = [torch.from_numpy(arr) for arr in train_target_as_list_of_np]

        criterion = nn.MSELoss()
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8)

        n_steps = 10
        for i in range(n_steps):
            print("Step", i)

            def closure():
                optimizer.zero_grad()
                out_list = [self.model.forward(tens[None, :]) for tens in train_input_list]
                loss = sum([criterion(out[0], target) for out, target in zip(out_list, train_target_list)])
                print("loss", loss)
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
