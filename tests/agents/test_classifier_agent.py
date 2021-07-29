import os
import logging
import unittest
import torch as T
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from alchemist.agents.classifier_agent import ClassifierAgent as Agent
from alchemist.data.data_utils import *

# NOTE: This is probably the wrong way to test a CNN, and may need rewriting

class TestAgentCreation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Using fake data, this class isn't for specifically crypto or tickers
        x = [[[3.0,3.0,3.0], [3.0,3.0,3.0]]] * 50
        y = [1] * 50
        x += [[[-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0]]] * 50
        y += [0] * 50
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=0.8, shuffle=True)
        cls.train_ds = Dataset(x_train, y_train)
        cls.test_ds = Dataset(x_test, y_test)

    def test_basic_initialization(self):
        # We should be able to initialize the agent easily, providing only
        # the training data in the form of a dataset
        agent = Agent(self.test_ds)
        self.assertEqual(agent.n_features, 2)
        self.assertIsInstance(agent.conv1, nn.Conv2d)
        self.assertIsInstance(agent.maxpool1, nn.MaxPool2d)
        self.assertIsInstance(agent.fc1, nn.Linear)
        self.assertIsInstance(agent.device, T.device)
        self.assertIsNotNone(type(agent.optimizer))
        self.assertIsNotNone(type(agent.loss))

    def test_calc_input_dims(self):
        x = [[[3.0,3.0,3.0]]] * 50
        y = [1] * 50
        dataset1 = Dataset(x, y)
        x = [[[3.0,3.0,3.0], [3.0,3.0,3.0], [3.0,3.0,3.0]]] * 50
        y = [1] * 50
        dataset2 = Dataset(x, y)
        dims1 = Agent(dataset1).input_dims
        dims2 = Agent(dataset2).input_dims
        self.assertIsInstance(dims1, int)
        self.assertNotEqual(dims1, dims2)

    def test_train_test_data_loaders(self):
        agent = Agent(self.train_ds)
        self.assertIsInstance(agent.train_data_loader, T.utils.data.DataLoader)
        self.assertIsNone(agent.test_data_loader)
        agent = Agent(self.train_ds, self.test_ds)
        self.assertIsInstance(agent.train_data_loader, T.utils.data.DataLoader)
        self.assertIsInstance(agent.test_data_loader, T.utils.data.DataLoader)

    def test_initialization_using_baktest_ds(self):
        # Supplying only the backtest_ds should be enough to make the agent
        x = [[[[3.0,3.0,3.0], [3.0,3.0,3.0]], 
              [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]]] * 50
        y = [[1.2, 0.8]] * 50
        backtest_ds = Dataset(x, y)
        agent = Agent(backtest_ds = backtest_ds)
        self.assertIsNone(agent.train_data_loader)
        self.assertIsNone(agent.test_data_loader)
        self.assertEqual(agent.n_features, 2)
        self.assertIsInstance(agent.conv1, nn.Conv2d)
        self.assertIsInstance(agent.maxpool1, nn.MaxPool2d)
        self.assertIsInstance(agent.fc1, nn.Linear)
        self.assertIsInstance(agent.device, T.device)

    def test_changing_data_loader_params(self):
        agent = Agent(self.train_ds, num_workers = 5)
        self.assertEqual(agent.train_data_loader.num_workers, 5)
        agent = Agent(self.train_ds, self.test_ds, num_workers = 10)
        self.assertEqual(agent.train_data_loader.num_workers, 10)
        self.assertEqual(agent.test_data_loader.num_workers, 10)


class TestAgentFunctionality(unittest.TestCase):

    def setUp(self):
        # This fake data is easily classifyiable to test the agent easily
        x = [[[3.0,3.0,3.0], [3.0,3.0,3.0]]] * 50
        y = [1] * 50
        x += [[[-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0]]] * 50
        y += [0] * 50
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=0.8, shuffle=True)
        self.train_ds = Dataset(x_train, y_train)
        self.test_ds = Dataset(x_test, y_test)
        self.agent = Agent(self.train_ds, self.test_ds)

    def test_padding(self):
        x = T.zeros((1, 1, 3, 2)).to(self.agent.device)
        x = self.agent.conv1(x)
        x = self.agent.maxpool1(x)
        x = self.agent.conv2(x)
        x = self.agent.maxpool2(x)
        x = self.agent.conv3(x)
        x = self.agent.maxpool3(x)
        self.assertEqual(x.size(), T.Size([1, 32, 3, 2]))

    def test_forward(self):
        x = T.zeros((1, 1, 3, 2))
        y = self.agent.forward(x)
        self.assertEqual(y.size(), T.Size([1, 2]))
        self.assertEqual(y.device, self.agent.device)

    def test_save_load_agent(self):
        # NOTE: Maybe the state dict saved should be tested more thoroughly
        path = "cache/tests/agents/ClassifierAgentTestSave"
        if os.path.isfile(path):
            os.remove(path)
        self.agent.save_chkpt(path = path)
        # Check the save file has been made, try loading from it
        self.assertTrue(os.path.isfile(path))
        self.agent.load_chkpt(path = path)

    def test_train(self):
        # Run the train function
        train_hist = self.agent.train_(epochs = 30)
        # The acc and loss history should show improvement
        self.assertIsInstance(train_hist, pd.DataFrame)
        epochs = train_hist["epoch"]
        acc_h = train_hist["acc"].values
        loss_h = train_hist["loss"].values
        self.assertTrue(len(acc_h) > 0)
        self.assertTrue(len(loss_h) > 0)
        self.assertEqual(acc_h[-1], 1)
        self.assertAlmostEqual(loss_h[-1], 0, 1)

    def test_test(self):
        # Test an untrained agent
        acc1, loss1 = self.agent.test_()
        self.assertIsInstance(acc1, np.float64)
        # Test a trained agent
        self.agent.train_(epochs = 20)
        acc2, loss2 = self.agent.test_()
        # Compare
        self.assertTrue(acc1 < acc2)
        self.assertTrue(loss1 > loss2)

    def test_backtest(self):
        # This fake data should give predictable backtest results
        x = [[[[3.0,3.0,3.0], [3.0,3.0,3.0]], 
              [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]]] * 10
        y = [[1.2, 0.8]] * 10
        backtest_ds = Dataset(x, y)
        agent = Agent(self.train_ds, backtest_ds = backtest_ds)
        agent.train_(epochs = 10)
        backtest_results = agent.backtest()
        # The backtest results should contain the test's history for graphing,
        # the average daily earnings and total earnings
        self.assertIsInstance(backtest_results[0], list)
        self.assertAlmostEqual(backtest_results[1], 1.2)
        self.assertAlmostEqual(backtest_results[2], pow(1.2, 10))


if __name__ == "__main__":
    unittest.main()
