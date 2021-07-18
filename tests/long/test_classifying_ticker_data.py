import math
import unittest
import torch as T
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
        print("Basic Test : single ticker")
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
        print("Basic Test : multiple tickers")
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
        print("Basic Test : hard-split train and test data")
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
        print("Hyperparameter Tuning : control")
        # A control is needed to compare the tests to
        train_hist = self.agent._train(epochs = 300, verbose = True)
        acc, loss = self.agent._test(verbose = True)

        save_data(train_hist, "cache/tests/data/long_test_results/control")
        path = "cache/tests/plots/ticker_agent_testing/hyperparameter_tuning/control"
        graph_train_data(data = train_hist, path = path, step = 5)

    def test_with_increased_features(self):
        print("Hyperparameter Tuning : more features")
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
        print("Hyperparameter Tuning : adjusted volatility")
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
        print("Hyperparameter Tuning : adjusted volatility and lowered lr")
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

class TestActualGains(unittest.TestCase):
    
    def setUp(self):
        self.path = "cache/tests/data/TestNewAgentForActualGains"
        self.raw_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                        from_date = "2010-01-01", to_date = "2021-01-01")
        # self.data = format_into_percentages(self.raw_data, formatting_basis = "daily close")
        # x, y = format_into_xy(self.data, num_features = 5, label_type = "bin", 
                # label_type_vars = {"divider" : 1, "balance" : True})
        # train_ds, test_ds = train_test_datasets(x, y)

        # self.agent = Agent(train_ds, test_ds)

    def backtest(self, agent, x, y, n_tickers = 1, backtest_name = ""):
        # x data should be formatted for the agent, y data only into percentages
        # print(x)
        x = T.tensor(x, dtype = T.float32)
        x = T.unsqueeze(x, 1)
        prediction = agent.forward(x).cpu()
        # print(prediction)
        prediction = F.softmax(prediction, dim=1)
        # print(prediction)
        choices = T.argmax(prediction, dim=1)
        choices = list(choices)
        
        choices = [list(j) for j in np.array_split(choices, n_tickers)]
        choices = list(map(list, zip(*choices)))
        y = [list(j) for j in np.array_split(y, n_tickers)]
        y = list(map(list, zip(*y)))
        # gains = [(y[i] if choices[i] == 1 else 1) for i in range(len(y[0]))]
        # gains = [((np.sum([y[j][i] if choices[j][i] == 1 else 1]) - [choices[j][i]].count(1))
            # /(n_tickers-[choices[j][i]].count(1)) for j in range(n_tickers)) for i in range(len(y[0]))]
        gains = [(np.sum([y[i][j] if choices[i][j] == 1 else 1 for j in range(n_tickers)]) 
            - choices[i].count(0))/(n_tickers-choices[i].count(0)) for i in range(len(y))]
        gains = [i if not math.isnan(i) else 1 for i in gains]
        # print(y[:5])
        # print(choices[:5])
        # print(gains[:5])

        total = np.product(gains)
        average = np.mean(gains)
        print("Total:", total, "or %.3f" % (total*100) + "%")
        print("Average:", average, "or %.3f" % ((average-1)*100) + "%")
        print("Days:", len(y))
        # print(choices.count(0))
        # print(gains.count(1))
        # print(gains)
        path = "cache/tests/plots/ticker_agent_testing/backtests/" + backtest_name + "backtest.png"
        graph_backtest_data(gains, path)

    def test_basic_backtest(self):
        print("Backtests : basic")
        # As seen several times above, but without a train-test split
        training_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                                 from_date = "2010-01-01", to_date = "2018-01-01")
        training_data = format_into_percentages(training_data, formatting_basis = "daily open")
        train_x, train_y = format_into_xy(training_data, num_features = 5, label_type = "bin",
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds = TickerDataset(train_x, train_y)
        agent = Agent(train_ds)
        agent._train(epochs = 1000, verbose = True)
        print("Training done!")

        # The test data is simple to format, thanks to 'label_type = "float"'
        test_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                             from_date = "2018-01-01", to_date = "2021-01-01")
        test_data = format_into_percentages(test_data, formatting_basis = "daily open")
        test_x, test_y = format_into_xy(test_data, num_features = 5, label_type = "float",
                label_type_vars = {"divider" : 1, "balance" : False})

        print("Beginning backtest...")
        # backtest does all the rest of the work
        self.backtest(agent, test_x, test_y, len(test_data["Data"].keys()))

    def test_volatility_backtest(self):
        print("Backtests : adjusted volatility")
        # As seen several times above, but without a train-test split
        training_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                                 from_date = "2010-01-01", to_date = "2018-01-01")
        training_data = format_into_percentages(training_data, formatting_basis = "daily open")
        training_data = adjust_for_volatility(training_data, volatility_type = "daily v")
        train_x, train_y = format_into_xy(training_data, num_features = 5, label_type = "bin",
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds = TickerDataset(train_x, train_y)
        agent = Agent(train_ds, learning_rate = 5e-3)
        agent._train(epochs = 1000, verbose = True)
        print("Training done!")

        # The test data is simple to format, thanks to 'label_type = "float"'
        test_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                             from_date = "2018-01-01", to_date = "2021-01-01")
        test_data = format_into_percentages(test_data, formatting_basis = "daily open")
        not_x, test_y = format_into_xy(test_data, num_features = 5, label_type = "float",
                label_type_vars = {"divider" : 1, "balance" : False})
        test_data = adjust_for_volatility(test_data, volatility_type = "daily v")
        test_x, not_y = format_into_xy(test_data, num_features = 5, label_type = "float",
                label_type_vars = {"divider" : 1, "balance" : False})

        print("Beginning backtest...")
        # backtest does all the rest of the work
        self.backtest(agent, test_x, test_y, len(test_data["Data"].keys()), backtest_name = "adjusted_volatility_")

    def test_backtest_with_more_features(self):
        print("Backtests : increased features")
        # As seen several times above, but without a train-test split
        training_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                                 from_date = "2010-01-01", to_date = "2018-01-01")
        training_data = format_into_percentages(training_data, formatting_basis = "daily open")
        train_x, train_y = format_into_xy(training_data, num_features = 30, label_type = "bin",
                label_type_vars = {"divider" : 1, "balance" : True})
        train_ds = TickerDataset(train_x, train_y)
        agent = Agent(train_ds)
        agent._train(epochs = 500, verbose = True)
        print("Training done!")

        # The test data is simple to format, thanks to 'label_type = "float"'
        test_data = get_data(path = self.path, tickers = ["AAPL", "GOOG", "AMZN"], 
                             from_date = "2018-01-01", to_date = "2021-01-01")
        test_data = format_into_percentages(test_data, formatting_basis = "daily open")
        test_x, test_y = format_into_xy(test_data, num_features = 30, label_type = "float",
                label_type_vars = {"divider" : 1, "balance" : False})

        print("Beginning backtest...")
        # backtest does all the rest of the work
        self.backtest(agent, test_x, test_y, len(test_data["Data"].keys()), backtest_name = "increased_features_")

