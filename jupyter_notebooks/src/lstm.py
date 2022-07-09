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
from torch.autograd import Variable

"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""


class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=32, number_of_layers=6):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm_list = [nn.LSTMCell(1, n_hidden).double()]
        self.lstm_list += [nn.LSTMCell(n_hidden, n_hidden).double() for _ in range(number_of_layers - 1)]
        assert len(self.lstm_list) == number_of_layers
        self.linear = nn.Linear(self.n_hidden, 1).double()

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)
        hti_cti = [
            (
                Variable(torch.zeros(n_samples, self.n_hidden, dtype=torch.float64)),
                Variable(torch.zeros(n_samples, self.n_hidden, dtype=torch.float64))
            )
            for _ in self.lstm_list
        ]
        assert len(hti_cti) == len(self.lstm_list)
        for input_t in x.split(1, dim=1):
            for i in range(len(self.lstm_list)):
                hti_cti[i] = self.lstm_list[i](input_t if i == 0 else hti_cti[i-1][0], hti_cti[i])
            output = self.linear(hti_cti[-1][0])
            outputs += [output]
        for _ in range(future):  # if we should predict the future
            for i in range(len(self.lstm_list)):
                hti_cti[i] = self.lstm_list[i](output if i == 0 else hti_cti[i-1][0], hti_cti[i])
            output = self.linear(hti_cti[-1][0])
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
        training_data_as_list_of_tensors = [Variable(torch.from_numpy(arr)) for arr in padded_training_data_set_as_list_of_np]
        big_tensor = Variable(torch.stack(training_data_as_list_of_tensors))
        return big_tensor

    def learn_from_data_set(self, training_data_set):
        pad = -2
        big_tensor = self.__turn_training_data_set_into_one_big_tensor(
            training_data_set=training_data_set,
            number_to_fill_empty_with=pad
        )
        train_input = big_tensor[:, :-1]
        train_target = big_tensor[:, 1:]
        true_if_pad = train_target == pad
        false_if_pad = train_target != pad

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8)

        n_steps = 10
        for i in range(n_steps):
            print("Step", i)

            def closure():
                optimizer.zero_grad()
                out = self.model.forward(train_input)
                loss_array = criterion(out, train_target)
                loss_array[true_if_pad] = 0
                loss = loss_array.sum() / false_if_pad.sum()
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
            res = y[0][-how_much_to_predict:]
            assert isinstance(res, np.ndarray)
            assert len(res) == how_much_to_predict
            assert res.shape == (how_much_to_predict,)
            assert res.dtype == np.float64
            return res


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
