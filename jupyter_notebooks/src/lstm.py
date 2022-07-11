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
import time
import os

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
        return out


"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size=64, num_layers=5):
        super(LSTMPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1,
            ),
            ExtractTensorAfterLSTM(),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=1
            )
        )

    def forward(self, x, future=0):
        out = None
        for i in range(future + 1):
            out = self.model(x)
            last_sample_in_each_series = out[:, -1, None, :]
            assert last_sample_in_each_series.shape == (x.size(0), 1, x.size(2))
            next_x = torch.cat((x, last_sample_in_each_series), dim=1)
            x = next_x
        return out


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
        self.batch_size = 32
        print("batch_size =", self.batch_size)
        self.num_epochs = 100
        print("num_epochs =", self.num_epochs)
        self.learning_rate = 0.002
        print("learning_rate =", self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')
        print("criterion =", self.criterion)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print("optimizer =", self.optimizer)

    @staticmethod
    def __torch_from_numpy(array):
        return torch.from_numpy(array).to(torch.float32)

    @staticmethod
    def __get_data_as_list_of_np_arrays(training_data_set):
        training_data_set_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        return training_data_set_as_list_of_np

    def __save_model(self):
        is_saved = False
        for path in [f"./models/", f"../models/"]:
            if os.path.exists(path):
                assert not is_saved
                torch.save(self.model.state_dict(), f"{path}lstm.pt")
                is_saved = True
        assert is_saved

    @staticmethod
    def __does_save_exist():
        return os.path.exists(f"./models/lstm.pt") or os.path.exists(f"../models/lstm.pt")

    def __load_model(self):
        is_loaded = False
        for path in [f"./models/", f"../models/"]:
            if os.path.exists(path):
                assert not is_loaded
                self.model.load_state_dict(torch.load(f"{path}lstm.pt"))
                model.eval()
                is_loaded = True
        assert is_loaded

    def __partition_list_to_batches(self, list_of_np_array):
        random.shuffle(list_of_np_array)
        num_batches = math.ceil(len(list_of_np_array) / self.batch_size)
        result = [
            list_of_np_array[i * self.batch_size: (i + 1) * self.batch_size]
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
            Variable(self.__torch_from_numpy(arr)[:, None]) for arr in padded_batch_as_list_of_np
        ]
        batch_tensor = Variable(torch.stack(batch_as_list_of_tensors))
        predict_length = 0
        train_input = batch_tensor[:, :-1]
        train_target = batch_tensor[:, 1:]
        true_if_pad = (train_target == self.pad)
        false_if_pad = (train_target != self.pad)
        return train_input, train_target, true_if_pad, false_if_pad, predict_length

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

    def __do_batch(self, batch_data):
        (train_input, train_target, true_if_pad, false_if_pad, predict_length) = batch_data
        self.optimizer.zero_grad()
        out = self.model.forward(train_input, future=predict_length)
        loss_array = self.criterion(out, train_target)
        loss_array[true_if_pad] = 0
        loss = loss_array.sum() / false_if_pad.sum()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __do_epoch(self, epoch_num, list_of_batch, training_data_set):
        sum_of_losses = 0
        for i, batch_data in enumerate(list_of_batch):
            # print(f"Batch {i+1} / {len(list_of_batch)}")
            loss = self.__do_batch(batch_data=batch_data)
            print(f"loss of batch {i + 1} / {len(list_of_batch)}: {loss}")
            sum_of_losses += loss
        # choose random sample and plot
        self.__plot_prediction_of_random_sample(training_data_set=training_data_set)
        if epoch_num % 10 == 0:
            self.__save_model()
        return sum_of_losses

    def __do_training(self, training_data_set):
        list_of_np_array = self.__get_data_as_list_of_np_arrays(
            training_data_set=training_data_set,
        )
        list_of_batch = self.__list_of_np_array_to_list_of_batch(
            list_of_np_array=list_of_np_array
        )
        epoch_time = 0
        for e in range(self.num_epochs):
            print(f"Epoch {e + 1} / {self.num_epochs}. Last epoch time was {epoch_time}")
            epoch_start_time = time.time()
            sum_of_losses = self.__do_epoch(
                epoch_num=e,
                list_of_batch=list_of_batch,
                training_data_set=training_data_set
            )
            epoch_stop_time = time.time()
            epoch_time = epoch_stop_time - epoch_start_time
            avg_loss = sum_of_losses / len(list_of_batch)
            print(f"************************ Average loss for the batches in the epoch: {avg_loss}")

    def learn_from_data_set(self, training_data_set):
        if self.__does_save_exist():
            self.__load_model()
        else:
            self.__do_training(training_data_set=training_data_set)

    def predict(self, ts_as_df_start, how_much_to_predict):
        with torch.no_grad():
            ts_as_np = ts_as_df_start["sample"].to_numpy()
            ts_as_tensor = self.__torch_from_numpy(ts_as_np)
            prediction = self.model.forward(ts_as_tensor[None, :, None], future=how_much_to_predict)
            prediction_flattened = prediction.view(len(ts_as_np) + how_much_to_predict)
            y = prediction_flattened.detach().numpy()
            res = np.float64(y[-how_much_to_predict:])
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
