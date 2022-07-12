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
        return out[:, -1, :]


"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=5):
        super(LSTMPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
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
                out_features=output_size
            )
        )
        self.output_size = output_size

    def forward(self, x, future=None):
        if future is None:
            future = self.output_size
        outs = []
        out = None
        for i in range(future - self.output_size + 1):
            out = self.model(x)
            last_sample_in_each_series_shaped_like_out = out[:, -1, None]
            outs += [last_sample_in_each_series_shaped_like_out]
            last_sample_in_each_series_shaped_like_x = out[:, -1, None, None]
            assert last_sample_in_each_series_shaped_like_x.shape == (x.size(0), 1, x.size(2))
            next_x = torch.cat((x, last_sample_in_each_series_shaped_like_x), dim=1)
            x = next_x
        if future == self.output_size:
            return out
        else:
            outs_as_tensor = torch.cat(outs[:-1], dim=1)
            result = torch.cat((outs_as_tensor, out), dim=1)
            return result

"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class LSTMTester:
    def __init__(self):
        self._msg = "[LSTM]"
        self.model = None
        # self.pad = -2
        # print(self._msg, f"pad = {self.pad}")
        self.batch_size = 128
        print(self._msg, f"batch_size =", self.batch_size)
        self.num_epochs = 30
        print(self._msg, f"num_epochs =", self.num_epochs)
        self.learning_rate = 0.0001
        print(self._msg, f"learning_rate =", self.learning_rate)
        # self.criterion = nn.MSELoss(reduction='none')
        self.criterion = nn.MSELoss()
        print(self._msg, f"criterion =", self.criterion)

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
        # TODO: change this
        # return os.path.exists(f"./models/lstm.pt") or os.path.exists(f"../models/lstm.pt")
        return False

    def __load_model(self):
        is_loaded = False
        for path in [f"./models/", f"../models/"]:
            if os.path.exists(path):
                assert not is_loaded
                self.model.load_state_dict(torch.load(f"{path}lstm.pt"))
                self.model.eval()
                is_loaded = True
        assert is_loaded

    def __partition_list_to_batches(self, list_of_something):
        random.shuffle(list_of_something)
        num_batches = math.ceil(len(list_of_something) / self.batch_size)
        result = [
            list_of_something[i * self.batch_size: (i + 1) * self.batch_size]
            for i in range(num_batches)
        ]
        return result

    def __turn_np_array_into_list_of_samples(self, np_array, input_size, output_size):
        list_of_input_output = [
            (np_array[i: input_size + i], np_array[input_size + i: output_size + input_size + i])
            for i in range(len(np_array) - input_size - output_size + 1)
        ]
        list_of_input_output_tensors = [
            (Variable(self.__torch_from_numpy(a)[:, None]), Variable(self.__torch_from_numpy(b)))
            for a, b in list_of_input_output
        ]
        return list_of_input_output_tensors

    @staticmethod
    def __combine_batches(batches):
        combined_batches = []
        for batch in batches:
            # TODO: stack
            batch_in = [tup[0] for tup in batch]
            batch_out = [tup[1] for tup in batch]
            combined_batches += [(torch.stack(batch_in), torch.stack(batch_out))]
        return combined_batches

    def __list_of_np_array_to_list_of_batch(self, list_of_np_array):
        min_length = min([len(arr) for arr in list_of_np_array])
        input_size = min_length // 2
        output_size = min_length - input_size
        self.__make_model_using_smallest_time_series_size(
            input_size=1,
            output_size=output_size
        )
        list_of_samples = []
        for np_arr in list_of_np_array:
            list_of_samples += self.__turn_np_array_into_list_of_samples(
                np_array=np_arr,
                input_size=input_size,
                output_size=output_size
            )
        print(self._msg, f"Length of list_of_samples = ", len(list_of_samples))
        batches = self.__partition_list_to_batches(
            list_of_something=list_of_samples
        )
        combined = self.__combine_batches(batches=batches)
        return combined

    def __plot_prediction_of_random_sample(self, training_data_set):
        print(self._msg, f"Plotting prediction for some random sample in the test set.")
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
        print(self._msg, f"MSE of this prediction is: {mse_here}")

    def __do_batch(self, batch_data):
        (train_input, train_target) = batch_data
        self.optimizer.zero_grad()
        out = self.model.forward(train_input)
        loss = self.criterion(out, train_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __do_epoch(self, epoch_num, list_of_batch, training_data_set):
        sum_of_losses = 0
        for i, batch_data in enumerate(list_of_batch):
            # print(f"Batch {i+1} / {len(list_of_batch)}")
            loss = self.__do_batch(batch_data=batch_data)
            print(self._msg, f"loss of batch {i + 1} / {len(list_of_batch)}: {loss}")
            sum_of_losses += loss
        # choose random sample and plot
        self.__plot_prediction_of_random_sample(training_data_set=training_data_set)
        if epoch_num % 10 == 0:
            self.__save_model()
        return sum_of_losses

    def __make_model_using_smallest_time_series_size(self, input_size, output_size):
        self.model = LSTMPredictor(
            input_size=input_size,
            output_size=output_size
        )
        print(self._msg, f"model = {self.model}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(self._msg, f"optimizer =", self.optimizer)

    def __do_training(self, training_data_set):
        list_of_np_array = self.__get_data_as_list_of_np_arrays(
            training_data_set=training_data_set,
        )
        list_of_batch = self.__list_of_np_array_to_list_of_batch(
            list_of_np_array=list_of_np_array
        )
        epoch_time = 0
        for e in range(self.num_epochs):
            print(self._msg, f"Epoch {e + 1} / {self.num_epochs}. Last epoch time was {epoch_time}")
            epoch_start_time = time.time()
            sum_of_losses = self.__do_epoch(
                epoch_num=e,
                list_of_batch=list_of_batch,
                training_data_set=training_data_set
            )
            epoch_stop_time = time.time()
            epoch_time = epoch_stop_time - epoch_start_time
            avg_loss = sum_of_losses / len(list_of_batch)
            print(self._msg, f"************************ Average loss for the batches in the epoch: {avg_loss}")

    def learn_from_data_set(self, training_data_set):
        if self.__does_save_exist():
            print(self._msg, f"Loading model instead of training because lstm.pt exists")
            self.__load_model()
        else:
            self.__do_training(training_data_set=training_data_set)

    def predict(self, ts_as_df_start, how_much_to_predict):
        with torch.no_grad():
            ts_as_np = ts_as_df_start["sample"].to_numpy()
            ts_as_tensor = self.__torch_from_numpy(ts_as_np)
            prediction = self.model.forward(ts_as_tensor[None, :, None], future=how_much_to_predict)
            prediction_flattened = prediction.view(how_much_to_predict)
            y = prediction_flattened.detach().numpy()
            res = np.float64(y)
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
