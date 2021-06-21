import unittest
import torch as T
import torch.nn as nn
import torch.optim as optim
from alchemist.data.ticker_data import *
from alchemist.agents.new_agent import *

# The new agent should have the brain of the old one, with improvements
# on how it interfaces with the rest of the code; these tests define this,
# and should be appliccable to any agent buit to replace the new agent

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
        data = get_data(path = "cache/tests/TestAgentFunctionality", tickers = ["GME", "TSLA"], 
                        from_date = "2020-01-01", to_date = "2021-01-01")
        data = format_into_percentages(data, formatting_basis = "daily open")
        x, y = format_into_xy(data, num_features = 5)
        # In some cases we don't need to test the agent, and provide a single ds
        self.dataset = TickerDataset(x, y)
        # In other cases we provide both a train and a test dataset
        self.train_ds, self.test_ds = train_test_datasets(x, y)

    def test_forward(self):
        pass

    def test_train(self):
        pass

    def test_test(self):
        pass
    
    def test_save_agent(self):
        pass

    def test_load_agent(self):
        pass


