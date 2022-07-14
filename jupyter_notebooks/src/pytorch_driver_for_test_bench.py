"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import random
import math
import test_bench
import time
import os

"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class PytorchTester:
    def __init__(self):
        # not accessible to owning class
        self.__msg = "[PytorchTester]"
        self.__best_model = None
        self.__criterion = nn.MSELoss(reduction='none')
        print(self.__msg, f"criterion =", self.__criterion)
        # accessible to owning class
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.batch_size = None
        self.padding = None
        self.num_epochs = None

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    @staticmethod
    def __torch_from_numpy(array):
        return torch.from_numpy(array).to(torch.float32)

    @staticmethod
    def __get_data_as_list_of_np_arrays(training_data_set):
        training_data_set_as_list_of_np = [ts_as_df["sample"].to_numpy() for ts_as_df in training_data_set]
        return training_data_set_as_list_of_np

    def __partition_list_to_batches(self, list_of_something):
        random.shuffle(list_of_something)
        num_batches = math.ceil(len(list_of_something) / self.batch_size)
        result = [
            list_of_something[i * self.batch_size: (i + 1) * self.batch_size]
            for i in range(num_batches)
        ]
        return result

    def __turn_np_array_into_list_of_samples(self, np_array):
        input_size = len(np_array) // 2
        output_size = len(np_array) - input_size
        list_of_input_output = []
        for k in range(self.sample_multiplier):
            list_of_input_output += [
                (np_array[i: input_size + i], np_array[input_size + i: output_size + input_size + i])
                for i in range(len(np_array) - input_size - output_size + 1)
            ]
            input_size -= (k + 1) % 2
            output_size -= k % 2
        list_of_input_output_tensors = [
            (Variable(self.__torch_from_numpy(a)[:, None]), Variable(self.__torch_from_numpy(b)))
            for a, b in list_of_input_output
        ]
        return list_of_input_output_tensors

    def __combine_batches(self, batches):
        combined_batches = []
        for batch in batches:
            # TODO: stack
            batch_in = [tup[0] for tup in batch]
            batch_out = [tup[1] for tup in batch]
            batch_in_tensor = torch.nn.utils.rnn.pad_sequence(
                batch_in, batch_first=True, padding_value=self.padding
            ).to(device=self.device)
            batch_out_tensor = torch.nn.utils.rnn.pad_sequence(
                batch_out, batch_first=True, padding_value=self.padding
            ).to(device=self.device)
            true_if_pad = (batch_out_tensor == self.padding)
            false_if_pad = (batch_out_tensor != self.padding)
            predict_length = batch_out_tensor.size(1)
            combined_batches += [(batch_in_tensor, batch_out_tensor, true_if_pad, false_if_pad, predict_length)]
        return combined_batches

    def __list_of_np_array_to_list_of_batch(self, list_of_np_array):
        list_of_samples = []
        for np_arr in list_of_np_array:
            list_of_samples += self.__turn_np_array_into_list_of_samples(
                np_array=np_arr,
            )
        print(self.__msg, f"Length of list_of_samples = ", len(list_of_samples))
        batches = self.__partition_list_to_batches(
            list_of_something=list_of_samples
        )
        combined = self.__combine_batches(batches=batches)
        return combined

    def __plot_prediction_of_random_sample(self, training_data_set):
        print(self.__msg, f"Plotting prediction for some random sample in the test set.")
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
        print(self.__msg, f"MSE of this prediction is: {mse_here}")

    def __do_batch(self, batch_data):
        train_input, train_target, true_if_pad, false_if_pad, predict_length = batch_data
        self.optimizer.zero_grad()
        out = self.model.forward(x=train_input, future=predict_length)
        loss_array = self.__criterion(out, train_target)
        loss_array[true_if_pad] = 0
        loss = loss_array.sum() / false_if_pad.sum()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __do_epoch(self, epoch_num, list_of_batch, training_data_set):
        sum_of_losses = 0
        for i, batch_data in enumerate(list_of_batch):
            loss = self.__do_batch(batch_data=batch_data)
            print(self.__msg, f"loss of batch {i + 1} / {len(list_of_batch)}: {loss}")
            sum_of_losses += loss
        # choose random sample and plot
        self.__plot_prediction_of_random_sample(training_data_set=training_data_set)
        return sum_of_losses

    def __do_training(self, training_data_set):
        list_of_np_array = self.__get_data_as_list_of_np_arrays(
            training_data_set=training_data_set,
        )
        list_of_batch = self.__list_of_np_array_to_list_of_batch(
            list_of_np_array=list_of_np_array
        )
        epoch_time = 0
        min_sum_of_losses = float('inf')
        for e in range(self.num_epochs):
            print(self.__msg, f"Epoch {e + 1} / {self.num_epochs}. Last epoch time was {epoch_time}")
            epoch_start_time = time.time()
            sum_of_losses = self.__do_epoch(
                epoch_num=e,
                list_of_batch=list_of_batch,
                training_data_set=training_data_set
            )
            if sum_of_losses < min_sum_of_losses:
                min_sum_of_losses = sum_of_losses
                self.__best_model = copy.deepcopy(self.model)
                assert not (self.__best_model is self.model)  # assert different objects
            epoch_stop_time = time.time()
            epoch_time = epoch_stop_time - epoch_start_time
            avg_loss = sum_of_losses / len(list_of_batch)
            print(self.__msg, f"************************ Average loss for the batches in the epoch: {avg_loss}")

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def learn_from_data_set(self, training_data_set):
        self.__do_training(training_data_set=training_data_set)

    def predict(self, ts_as_df_start, how_much_to_predict):
        with torch.no_grad():
            ts_as_np = ts_as_df_start["sample"].to_numpy()
            ts_as_tensor = self.__torch_from_numpy(ts_as_np)
            model = self.model if self.__best_model is None else self.__best_model
            prediction = model.forward(
                ts_as_tensor[None, :, None].to(device=self.device),
                future=how_much_to_predict
            )
            prediction_flattened = prediction.view(how_much_to_predict).cpu()
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
    pass


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
