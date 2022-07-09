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
import random
import math
import test_bench

"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,

            ).double(),
            'linear': nn.Linear(
                in_features=hidden_size,
                out_features=1
            ).double()
        })

    def forward(self, x, future=0):
        assert len(x.shape) == 3
        out = None
        for i in range(future + 1):
            out, _ = self.model['lstm'](x)
            out = self.model['linear'](out)
            last_sample_in_each_series = out[:, -1, None, :]
            assert last_sample_in_each_series.shape == (x.size(0), 1, x.size(2))
            next_x = torch.cat((x, last_sample_in_each_series), dim=1)
            x = next_x
        return out
        # outputs = []
        # n_samples = x.size(0)
        # hti_cti = [
        #     (
        #         Variable(torch.zeros(n_samples, self.n_hidden, dtype=torch.float64)),
        #         Variable(torch.zeros(n_samples, self.n_hidden, dtype=torch.float64))
        #     )
        #     for _ in self.lstm_list
        # ]
        # assert len(hti_cti) == len(self.lstm_list)
        # for input_t in x.split(1, dim=1):
        #     for i in range(len(self.lstm_list)):
        #         hti_cti[i] = self.lstm_list[i](input_t if i == 0 else hti_cti[i-1][0], hti_cti[i])
        #     output = self.linear(hti_cti[-1][0])
        #     outputs += [output]
        # for _ in range(future):  # if we should predict the future
        #     for i in range(len(self.lstm_list)):
        #         hti_cti[i] = self.lstm_list[i](output if i == 0 else hti_cti[i-1][0], hti_cti[i])
        #     output = self.linear(hti_cti[-1][0])
        #     outputs += [output]
        # outputs = torch.cat(outputs, dim=1)
        # return outputs


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class LSTMTester:
    def __init__(self):
        self.model = LSTMPredictor()
        print(self.model)
        self.pad = -2
        print("pad =", self.pad)
        self.batch_size = 100
        print("batch_size =", self.batch_size)
        self.num_epochs = 1000
        print("num_epochs =", self.num_epochs)
        self.learning_rate = 0.005
        print("learning_rate =", self.learning_rate)

    @staticmethod
    def __get_data_as_list_of_np_arrays(training_data_set):
        training_data_set_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        return training_data_set_as_list_of_np

    def __partition_list_to_batches(self, list_of_np_array):
        random.shuffle(list_of_np_array)
        num_batches = math.ceil(len(list_of_np_array) / self.batch_size)
        result = [
            list_of_np_array[i * self.batch_size: (i+1)*self.batch_size]
            for i in range(num_batches)
        ]
        return result

    def __prepare_batch(self, batch_as_list):
        max_length = max([len(arr) for arr in batch_as_list])
        padded_batch_as_list_of_np = [
            np.concatenate(
                (
                    arr,
                    np.full((max_length - len(arr),), self.pad)
                )
            )
            for arr in batch_as_list
        ]
        batch_as_list_of_tensors = [
            Variable(torch.from_numpy(arr)[:, None]) for arr in padded_batch_as_list_of_np
        ]
        batch_tensor = Variable(torch.stack(batch_as_list_of_tensors))
        train_input = batch_tensor[:, :-1]
        train_target = batch_tensor[:, 1:]
        true_if_pad = (train_target == self.pad)
        false_if_pad = (train_target != self.pad)
        return train_input, train_target, true_if_pad, false_if_pad

    def __list_of_np_array_to_list_of_batch(self, list_of_np_array):
        batches = self.__partition_list_to_batches(
            list_of_np_array=list_of_np_array
        )
        result = []
        for batch in batches:
            b = self.__prepare_batch(batch_as_list=batch)
            result += [b]
        return result

    def __plot_prediction_of_random_sample(self, training_data_set):
        print("Plotting prediction for some random sample in the test set.")
        test_sample = random.choice([ts for ts in training_data_set])
        how_much_to_give = len(test_sample) // 2
        how_much_to_predict = len(test_sample) - how_much_to_give
        returned_ts_as_np_array = self.predict(
            ts_as_df_start=test_sample[: how_much_to_give],
            how_much_to_predict=how_much_to_predict
        )
        test_bench.plot_result(
            original=test_sample,
            prediction_as_np_array=returned_ts_as_np_array,
        )
        out_should_be = test_sample["sample"].to_numpy()[how_much_to_give:]
        mse_here = (np.square(out_should_be - returned_ts_as_np_array)).mean()
        print(f"MSE of this prediction is: {mse_here}")

    def learn_from_data_set(self, training_data_set):
        list_of_np_array = self.__get_data_as_list_of_np_arrays(
            training_data_set=training_data_set,
        )
        list_of_batch = self.__list_of_np_array_to_list_of_batch(
            list_of_np_array=list_of_np_array
        )
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for e in range(self.num_epochs):
            print(f"Epoch: {e+1} / {self.num_epochs}")
            for i, (train_input, train_target, true_if_pad, false_if_pad) in enumerate(list_of_batch):
                optimizer.zero_grad()
                out = self.model.forward(train_input)
                loss_array = criterion(out, train_target)
                loss_array[true_if_pad] = 0
                loss = loss_array.sum() / false_if_pad.sum()
                print(f"loss of batch {i} / {len(list_of_batch)}: {loss.item()}")
                loss.backward()
                optimizer.step()
            # choose random sample and plot
            self.__plot_prediction_of_random_sample(training_data_set=training_data_set)

    def predict(self, ts_as_df_start, how_much_to_predict):
        with torch.no_grad():
            ts_as_np = ts_as_df_start["sample"].to_numpy()
            ts_as_tensor = torch.from_numpy(ts_as_np)
            prediction = self.model.forward(ts_as_tensor[None, :, None], future=how_much_to_predict)
            prediction_flattened = prediction.view(len(ts_as_np) + how_much_to_predict)
            y = prediction_flattened.detach().numpy()
            res = y[-how_much_to_predict:]
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
