import os
import unittest
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from alchemist.data.ticker_data import *
from alchemist.agents.new_agent import *

# The new agent should have the brain of the old one, with improvements
# on how it interfaces with the rest of the code; these tests define this,
# and should be appliccable to any agent buit to replace the new agent

# NOTE: Most of this is probably the wrong way to test a PyTorch CNN. However I
# haven't been able to find the "right" way, and it feels wrong to leave it untested

class TestAgentCreation(unittest.TestCase):

    def setUp(self):
        data = get_data(path = "cache/tests/TestAgentCreation", tickers = ["GME", "TSLA"], 
                        from_date = "2021-03-01", to_date = "2021-04-01")
        data = format_into_percentages(data, formatting_basis = "daily open")
        x, y = format_into_xy(data, num_features = 3)
        # In some cases we don't need to test the agent, and provide a single ds
        self.dataset = TickerDataset(x, y)
        # In other cases we provide both a train and a test dataset
        self.train_ds, self.test_ds = train_test_datasets(x, y)

    def test_basic_initialization(self):
        # We should be able to initialize the agent easily, providing only
        # the training data in the form of a dataset
        agent = Agent(self.dataset)
        self.assertEqual(agent.n_features, 3)
        self.assertEqual(type(agent.conv1), nn.Conv2d)
        self.assertEqual(type(agent.maxpool1), nn.MaxPool2d)
        self.assertEqual(type(agent.fc1), nn.Linear)
        self.assertEqual(type(agent.device), T.device)
        self.assertIsNotNone(type(agent.optimizer))
        self.assertIsNotNone(type(agent.loss))
        # NOTE: There's a bit of functionality here that I'm not sure how to test,
        # like "self.to(self.device)", but it should work fine

    def test_calc_input_dims(self):
        agent = Agent(self.dataset)
        dims1 = agent.calc_input_dims(40, 5)
        dims2 = agent.calc_input_dims(50, 6)
        self.assertEqual(type(dims1), int)
        self.assertNotEqual(dims1, dims2)

    def test_train_test_data_loaders(self):
        agent = Agent(self.train_ds)
        self.assertEqual(type(agent.train_data_loader), T.utils.data.DataLoader)
        self.assertIsNone(agent.test_data_loader)
        agent = Agent(self.train_ds, self.test_ds)
        self.assertEqual(type(agent.train_data_loader), T.utils.data.DataLoader)
        self.assertEqual(type(agent.test_data_loader), T.utils.data.DataLoader)


class TestAgentFunctionality(unittest.TestCase):

    def setUp(self):
        # Using really obvious, but still 2d data just to test everything works as it should
        x = [[[3.0,3.0,3.0], [3.0,3.0,3.0]]] * 50
        y = [1] * 50
        x += [[[-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0]]] * 50
        y += [0] * 50
        train_ds, test_ds = train_test_datasets(x, y)
        self.agent = Agent(train_ds, test_ds, batch_size = 8)

    def test_padding(self):
        x = T.zeros((1, 1, 3, 2))
        x = self.agent.conv1(x)
        x = self.agent.maxpool1(x)
        x = self.agent.conv2(x)
        x = self.agent.maxpool2(x)
        self.assertEqual(x.size(), T.Size([1, 32, 3, 2]))

    def test_forward(self):
        x = T.zeros((1, 1, 3, 2))
        y = self.agent.forward(x)
        self.assertEqual(y.size(), T.Size([1, 2]))

    def test_save_load_agent(self):
        # I'm not quite sure how to test state dicts etc, this shouls be fine
        path = "cache/tests/agents/test_load_save_agent"
        if os.path.isfile(path):
            os.remove(path)
        self.agent.save_chkpt(path = path)
        # Check the save file has been made, try loading from it
        self.assertTrue(os.path.isfile(path))
        self.agent.load_chkpt(path = path)

    def test_train(self):
        # Run the train function
        acc_h, loss_h = self.agent._train(epochs = 5)
        # The acc and loss history should show improvement
        self.assertTrue(len(acc_h) > 0)
        self.assertTrue(len(loss_h) > 0)
        self.assertEqual(np.mean(acc_h[-1]), 1)
        self.assertAlmostEqual(np.mean(loss_h[-1]), 0, 2)

    def test_test(self):
        # Test an untrained agent
        acc_hu, loss_hu = self.agent._test()
        # Test a trained agent
        self.agent._train(epochs = 5)
        acc_ht, loss_ht = self.agent._test()
        # Compare
        self.assertTrue(np.mean(acc_ht) > np.mean(acc_hu))
        self.assertTrue(np.mean(loss_ht) < np.mean(loss_hu))


