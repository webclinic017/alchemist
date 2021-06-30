import os
import random
import unittest
import pandas as pd
from tests.utils.graphing import *


class TestGraphing(unittest.TestCase):

    def test_graphing_train_data(self):
        acc_hist = range(0, 50)
        loss_hist = range(0, 500, 10)[::-1]
        epochs = range(1, 51)
        results_df = pd.DataFrame({"epoch" : epochs,
                                   "acc" : acc_hist,
                                   "loss" : loss_hist})
        path = "cache/tests/plots/graph_testing/test_graphing_train_data.png"
        if os.path.isfile(path): os.remove(path)
        graph_train_data(results_df, path = path)
        self.assertTrue(os.path.isfile(path))

    def test_graphing_dense_train_data(self):
        acc_hist = []
        for x in range(100): acc_hist.append(random.random())
        loss_hist = range(0, 100)[::-1]
        epochs = range(100)
        results_df = pd.DataFrame({"epoch" : epochs,
                                   "acc" : acc_hist,
                                   "loss" : loss_hist})
        path = "cache/tests/plots/graph_testing/test_graphing_dense_train_data.png"
        # step = 1 should only graph every 10th datapoint; good for dense data
        graph_train_data(results_df, path = path, step = 10)


