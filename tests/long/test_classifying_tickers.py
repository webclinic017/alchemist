import unittest
import torch as T
from tests.utils.graphing import *
from alchemist.data.data_utils import *
from alchemist.data.ticker_data import *
from alchemist.agents.new_agent import *

# The top ten weighted stocks of the NASDAQ 100
top_nasdaq_tickers = ["AAPL", "MSFT", "AMZN", "GOOG", #"FB", 
                      "GOOGL", "NVDA", "ADBE"] #, "TSLA", "PYPL"]

class Test1TrainAgents(unittest.TestCase):
    # Tests how effectively the agents train, 
    # and also trains and saves them for future tests

    def get_train_test_data(self, path="cache/tests/data/long/train_data", 
                            tickers=top_nasdaq_tickers,
                            from_date="2011-01-01", to_date="2018-01-01",
                            formatting_basis="daily close", num_features=5):
        data = get_data(path=path, tickers=tickers,
                        from_date=from_date, to_date=to_date)
        data = format_into_percentages(data, formatting_basis=formatting_basis)
        x, y = format_into_xy(data, num_features=num_features, label_type="bin",
                              balance=True)
        train_ds, test_ds = train_test_datasets(x, y)
        return train_ds, test_ds

    def test_control(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data()
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=500, verbose=True)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_control")
        save_data(train_hist, "cache/tests/results/long/control_training")

    def test_10_features(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(num_features=10)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=500, verbose=True)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_10_features")
        save_data(train_hist, "cache/tests/results/long/10_features_training")

    def test_30_features(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(num_features=30)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=500, verbose=True)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_30_features")
        save_data(train_hist, "cache/tests/results/long/30_features_training")

class Test2BacktestAgents(unittest.TestCase):
    # Tests the agents on real data from recent years 

    def get_test_data(self, path="cache/tests/data/long/backtest_data", 
                      tickers=top_nasdaq_tickers,
                      from_date="2018-01-01", to_date="2021-01-01",
                      formatting_basis="daily close", n_features=5):
        test_data = get_data(path=path, tickers=tickers,
                             from_date=from_date, to_date=to_date)
        test_data = format_into_percentages(
                test_data, formatting_basis=formatting_basis)
        test_x, test_y = format_into_xy(
                test_data, num_features=n_features, label_type="float",
                divider=1, balance=False)
        example_ds = TickerDataset(test_x, test_y)

        return (test_x, test_y, example_ds)

    def backtest(self, agent, x, y, n_tickers=len(top_nasdaq_tickers)):
        x = T.tensor(x, dtype = T.float32)
        x = T.unsqueeze(x, 1)
        prediction = agent.forward(x).cpu()
        prediction = F.softmax(prediction, dim=1)
        choices = T.argmax(prediction, dim=1)
        choices = list(choices)
        choices = [list(j) for j in np.array_split(choices, n_tickers)]
        choices = list(map(list, zip(*choices)))
        y = [list(j) for j in np.array_split(y, n_tickers)]
        y = list(map(list, zip(*y)))
        gains = [(np.sum([y[i][j] if choices[i][j] == 1 else 1 for j in range(n_tickers)]) 
            - choices[i].count(0))/(n_tickers-choices[i].count(0)) for i in range(len(y))]
        gains = [i if not math.isnan(i) else 1 for i in gains]
        # Print some useful info
        total = np.product(gains)
        average = np.mean(gains)
        print("Total:", total, "or %.3f" % (total*100) + "%")
        print("Average:", average, "or %.3f" % ((average-1)*100) + "%")
        print("Days:", len(y))

        return gains
    
    def test_control(self):
        print("Control Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_control")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/control_backtest")

    def test_10_features(self):
        print("10 featurees Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data(n_features=10)
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_10_features")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/10_features_backtest")

    def test_30_features(self):
        print("30 features Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data(n_features=30)
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_30_features")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/30_features_backtest")

class Test3GraphData(unittest.TestCase):
    # Graphs all the data aquired during testing

    def test_control(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/control_training")
        backtest_data = load_data("cache/tests/results/long/control_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_5_features", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_5_features")

    def test_10_features(self):
        # load the data
        train_hist = load_data("cache/tests/results/long/10_features_training")
        backtest_data = load_data("cache/tests/results/long/10_features_backtest")
        # graph the data
        graph_train_data(train_hist, "cache/plots/long/training_10_features", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_10_features")

    def test_30_features(self):
        # load the data
        train_hist = load_data("cache/tests/results/long/30_features_training")
        backtest_data = load_data("cache/tests/results/long/30_features_backtest")
        # graph the data
        graph_train_data(train_hist, "cache/plots/long/training_30_features", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_30_features")
