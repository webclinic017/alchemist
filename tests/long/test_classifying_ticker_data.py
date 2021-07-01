import unittest
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tests.utils.graphing import *
from alchemist.data.data_utils import *
from alchemist.data.ticker_data import *
from alchemist.agents.new_agent import *


class TestAgentBasic(unittest.TestCase):

    def test_single_ticker(self):
        # Just 10 years of apple data used here
        path = "cache/tests/data/test_new_agent_single_ticker_data_basic"
        data = get_data(path = path, tickers = ["AAPL"], 
                        from_date = "2005-01-01", to_date = "2021-01-01")
        data = format_into_percentages(data, formatting_basis = "daily close")
        x, y = format_into_xy(data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds, test_ds = train_test_datasets(x, y)
        # Some test data is used to verify generalisation
        agent = Agent(train_ds, test_ds)
        # 500 epochs is about the right amount here
        train_hist = agent._train(epochs = 500, verbose = True)
        acc, loss = agent._test(verbose = True)

        # Save the results in case they need to be graphed differently later
        save_data(train_hist, "cache/tests/data/long_test_results/single_ticker")
        # Plot the results
        plot_path = "cache/tests/plots/ticker_agent_testing/single_ticker"
        graph_train_data(data = train_hist, path = plot_path, step = 5)


    def test_multiple_ticker(self):
        # Smaller time frame, more tickers, same amount of data total
        path = "cache/tests/data/test_new_agent_multiple_ticker_data_basic"
        data = get_data(path = path, tickers = ["AAPL", "GOOG", "AMZN"], 
                        from_date = "2015-01-01", to_date = "2020-01-01")
        data = format_into_percentages(data, formatting_basis = "daily close")
        x, y = format_into_xy(data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds, test_ds = train_test_datasets(x, y)
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs = 500, verbose = True)
        acc, loss = agent._test(verbose = True)

        save_data(train_hist, "cache/tests/data/long_test_results/three_ticker")
        plot_path = "cache/tests/plots/ticker_agent_testing/three_ticker"
        graph_train_data(data = train_hist, path = plot_path, step = 5)

    def test_with_split_train_test_data(self):
        # To remove possibility that data overlap between train and test data
        # is the cause of successful tests, here we ensure there is 0 overlap 
        # between train and test data
        path = "cache/tests/data/TestNewAgentWithChangingHyperparameters"
        train_data = get_data(path = path, tickers = ["AAPL", "GOOG", "AMZN"], 
                              from_date = "2010-01-01", to_date = "2018-01-01")
        train_data = format_into_percentages(train_data, formatting_basis = "daily close")
        train_x, train_y = format_into_xy(train_data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds = TickerDataset(train_x, train_y)
        test_data = get_data(path = path, tickers = ["AAPL", "GOOG", "AMZN"], 
                              from_date = "2018-01-01", to_date = "2019-01-01")
        test_data = format_into_percentages(test_data, formatting_basis = "daily close")
        test_x, test_y = format_into_xy(test_data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        test_ds = TickerDataset(test_x, test_y)
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs = 300, verbose = True)
        acc, loss = agent._test(verbose = True)
        print("Please verify test accuracy is sufficiently close to final train accuracy")

        save_data(train_hist, "cache/tests/data/long_test_results/manually_split_data")
        plot_path = "cache/tests/plots/ticker_agent_testing/manually_split_data"
        graph_train_data(data = train_hist, path = plot_path, step = 5)

class TestWithChangingHyperparameters(unittest.TestCase):

    # NOTE: Formatting all this data actually takes a while, but considering how long
    # the tests themselves take it's negligible
    def setUp(self):
        path = "cache/tests/data/TestNewAgentWithChangingHyperparameters"
        self.raw_data = get_data(path = path, tickers = ["AAPL", "GOOG", "AMZN"], 
                        from_date = "2010-01-01", to_date = "2020-01-01")
        self.data = format_into_percentages(self.raw_data, formatting_basis = "daily close")
        x, y = format_into_xy(self.data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds, test_ds = train_test_datasets(x, y)

        self.agent = Agent(train_ds, test_ds)

    def test_control(self):
        # A control is needed to compare the tests to
        train_hist = self.agent._train(epochs = 300, verbose = True)
        acc, loss = self.agent._test(verbose = True)

        save_data(train_hist, "cache/tests/data/long_test_results/control")
        path = "cache/tests/plots/ticker_agent_testing/hyperparameter_tuning/control"
        graph_train_data(data = train_hist, path = path, step = 5)

    def test_with_increased_features(self):
        # Here the agent is given five times as many features to consider
        x, y = format_into_xy(self.data, num_features = 30, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds, test_ds = train_test_datasets(x, y)
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs = 300, verbose = True)
        acc, loss = agent._test(verbose = True)

        save_data(train_hist, "cache/tests/data/long_test_results/increased_features")
        path = "cache/tests/plots/ticker_agent_testing/hyperparameter_tuning/increased_features"
        graph_train_data(data = train_hist, path = path, step = 5)
    
    def test_with_adjusting_for_volatility(self):
        # Here we adjust for volatility to see if that makes a difference
        data = adjust_for_volatility(self.data, volatility_type = "daily v")
        x, y = format_into_xy(data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds, test_ds = train_test_datasets(x, y)
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs = 300, verbose = True)
        acc, loss = agent._test(verbose = True)

        save_data(train_hist, "cache/tests/data/long_test_results/volatility_adjusted")
        path = "cache/tests/plots/ticker_agent_testing/hyperparameter_tuning/volatility_adjusted"
        graph_train_data(data = train_hist, path = path, step = 5)

    def test_adjusting_for_volatility_with_lowered_lr(self):
        # Here we adjust for volatility to see if that makes a difference
        data = adjust_for_volatility(self.data, volatility_type = "daily v")
        x, y = format_into_xy(data, num_features = 5, label_type = "bin", 
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds, test_ds = train_test_datasets(x, y)
        agent = Agent(train_ds, test_ds, learning_rate = 6e-3)
        train_hist = agent._train(epochs = 300, verbose = True)
        acc, loss = agent._test(verbose = True)

        save_data(train_hist, "cache/tests/data/long_test_results/volatility_adjusted_lower_lr")
        path = "cache/tests/plots/ticker_agent_testing/hyperparameter_tuning/volatility_adjusted_lower_lr"
        graph_train_data(data = train_hist, path = path, step = 5)



