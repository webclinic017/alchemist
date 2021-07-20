import unittest
import torch as T
from tests.utils.graphing import *
from alchemist.data.data_utils import *
from alchemist.data.ticker_data import *
from alchemist.agents.new_agent import *

# The top ten weighted stocks of the NASDAQ 100
top_nasdaq_tickers = ["AAPL", "MSFT", "AMZN", "GOOG", #"FB", 
                      "GOOGL", "NVDA", "ADBE"] #, "TSLA", "PYPL"]
# top_nasdaq_tickers = ["AAPL", "GOOG", "AMZN"]

class Test1TrainAgents(unittest.TestCase):
    # Tests how effectively the agents train, 
    # and also trains and saves them for future tests

    def get_train_test_data(self, path="cache/tests/data/long/train_data", 
                            tickers=top_nasdaq_tickers,
                            from_date="2011-01-01", to_date="2018-01-01",
                            formatting_basis="daily close", num_features=5,
                            divider=1, offset=0):
        data = get_data(path=path, tickers=tickers,
                        from_date=from_date, to_date=to_date)
        data = format_into_percentages(data, formatting_basis=formatting_basis)
        x, y = format_into_xy(data, num_features=num_features, label_type="bin",
                              balance=True, divider=divider, offset=offset)
        train_ds, test_ds = train_test_datasets(x, y)
        return train_ds, test_ds

    def _control(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data()
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_control")
        save_data(train_hist, "cache/tests/results/long/control_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def test_offset_5(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(features=30, offset=5)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_offset_5")
        save_data(train_hist, "cache/tests/results/long/offset_5_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def test_offset_10(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(num_features=30, offset=10)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_offset_10")
        save_data(train_hist, "cache/tests/results/long/offset_10_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _volatility(self):
        # Get training data
        data = get_data(path="cache/tests/data/long/train_data", 
                        tickers=top_nasdaq_tickers,
                        from_date="2011-01-01", to_date="2018-01-01")
        data = adjust_for_volatility(data, volatility_type="daily v")
        x, y = format_into_xy(data, num_features=5, label_type="bin",
                              balance=True, divider=0.99)
        train_ds, test_ds = train_test_datasets(x, y)
        # Make and train the agent
        agent = Agent(train_ds, test_ds, learning_rate=3e-3)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_volatility")
        save_data(train_hist, "cache/tests/results/long/volatility_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _divider_11(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(divider=1.01)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_divider11")
        save_data(train_hist, "cache/tests/results/long/divider11_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _divider_13(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(divider=1.03)
        # Make and train the agent
        agent = Agent(train_ds, test_ds, learning_rate=4e-4)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_divider13")
        save_data(train_hist, "cache/tests/results/long/divider13_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _divider_15(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(divider=1.05)
        # Make and train the agent
        agent = Agent(train_ds, test_ds, learning_rate=2e-4)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_divider15")
        save_data(train_hist, "cache/tests/results/long/divider15_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _divider_102(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(divider=1.002)
        # Make and train the agent
        agent = Agent(train_ds, test_ds, learning_rate=1e-3)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_divider102")
        save_data(train_hist, "cache/tests/results/long/divider102_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _10_features(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(num_features=10)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_10_features")
        save_data(train_hist, "cache/tests/results/long/10_features_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def test_30_features(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data(num_features=30)
        # Make and train the agent
        agent = Agent(train_ds, test_ds)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_30_features")
        save_data(train_hist, "cache/tests/results/long/30_features_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _smaller_kernel(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data()
        # Make and train the agent
        agent = Agent(train_ds, test_ds, kernel_size=2)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_small_kernel")
        save_data(train_hist, "cache/tests/results/long/small_kernel_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)

    def _larger_kernel(self):
        # Get training data
        train_ds, test_ds = self.get_train_test_data()
        # Make and train the agent
        agent = Agent(train_ds, test_ds, kernel_size=4)
        train_hist = agent._train(epochs=700, verbose=False)
        # Record all of the collected data and save the agent
        agent.save_chkpt(path="cache/tests/agents/long/agent_large_kernel")
        save_data(train_hist, "cache/tests/results/long/large_kernel_training")
        # Run a simple test to check for overfitting
        acc, loss = agent._test(verbose = True)



class Test2BacktestAgents(unittest.TestCase):
    # Tests the agents on real data from recent years 

    def get_test_data(self, path="cache/tests/data/long/backtest_data", 
                      tickers=top_nasdaq_tickers,
                      from_date="2018-01-01", to_date="2021-01-01",
                      formatting_basis="daily close", n_features=5,
                      offset=0):
        test_data = get_data(path=path, tickers=tickers,
                             from_date=from_date, to_date=to_date)
        test_data = format_into_percentages(
                test_data, formatting_basis=formatting_basis)
        test_x, test_y = format_into_xy(
                test_data, num_features=n_features, label_type="float",
                balance=False, offset=offset)
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

    def _control(self):
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

    def test_offset_5(self):
        print("Offset 5 Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_offset_5")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/offset_5_backtest")

    def test_offset_10(self):
        print("Offset 10 Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_offset_10")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/offset_10_backtest")

    def _volatility(self):
        print("Volatility Backtest:")
        # Get the test data
        test_data = get_data(path="cache/tests/data/long/backtest_data", 
                             tickers=top_nasdaq_tickers,
                             from_date="2018-01-01", to_date="2021-01-01")
        vol_test_data = adjust_for_volatility(test_data, volatility_type="daily v")
        per_test_data = format_into_percentages(test_data, formatting_basis="daily close")
        test_x, y = format_into_xy(
                vol_test_data, num_features=5, label_type="float",
                balance=False)
        x, test_y = format_into_xy(
                per_test_data, num_features=5, label_type="float",
                balance=False)
        example_ds = TickerDataset(test_x, test_y)
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_volatility")
        # Perform the backtest
        gains = self.backtest(agent, test_x, test_y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/volatility_backtest")

    def _divider_11(self):
        print("Divider 1.01 Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_divider11")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/divider11_backtest")

    def _divider_13(self):
        print("Divider 1.03 Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_divider13")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/divider13_backtest")

    def _divider_15(self):
        print("Divider 1.05 Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_divider15")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/divider15_backtest")

    def _divider_102(self):
        print("Divider 1.002 Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds)
        agent.load_chkpt("cache/tests/agents/long/agent_divider102")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/divider102_backtest")

    def _10_features(self):
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

    def _smaller_kernel(self):
        print("Smaller Kernel Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds, kernel_size=2)
        agent.load_chkpt("cache/tests/agents/long/agent_small_kernel")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/small_kernel_backtest")

    def _larger_kernel(self):
        print("Larger Kernel Backtest:")
        # Get the test data
        x, y, example_ds = self.get_test_data()
        # Load the agent
        agent = Agent(example_ds, kernel_size=4)
        agent.load_chkpt("cache/tests/agents/long/agent_large_kernel")
        # Perform the backtest
        gains = self.backtest(agent, x, y)
        # Save all gathered data
        save_data(gains, fname="cache/tests/results/long/large_kernel_backtest")

class Test3GraphData(unittest.TestCase):
    # Graphs all the data aquired during testing

    def _control(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/control_training")
        backtest_data = load_data("cache/tests/results/long/control_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_control", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_control")

    def test_offset_5(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/offset_5_training")
        backtest_data = load_data("cache/tests/results/long/offset_5_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_offset_5", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_offset_5")

    def test_offset_10(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/offset_10_training")
        backtest_data = load_data("cache/tests/results/long/offset_10_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_offset_10", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_offset_10")

    def _volatility(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/volatility_training")
        backtest_data = load_data("cache/tests/results/long/volatility_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_volatility", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_volatility")

    def _divider_11(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/divider11_training")
        backtest_data = load_data("cache/tests/results/long/divider11_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_divider11", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_divider11")

    def _divider_13(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/divider13_training")
        backtest_data = load_data("cache/tests/results/long/divider13_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_divider13", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_divider13")

    def _divider_15(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/divider15_training")
        backtest_data = load_data("cache/tests/results/long/divider15_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_divider15", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_divider15")

    def _divider_102(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/divider102_training")
        backtest_data = load_data("cache/tests/results/long/divider102_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_divider102", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_divider102")

    def _10_features(self):
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

    def _smaller_kernel(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/small_kernel_training")
        backtest_data = load_data("cache/tests/results/long/small_kernel_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_small_kernel", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_small_kernel")

    def _larger_kernel(self):
        # Load the data
        train_hist = load_data("cache/tests/results/long/large_kernel_training")
        backtest_data = load_data("cache/tests/results/long/large_kernel_backtest")
        # Graph the data
        graph_train_data(train_hist, "cache/plots/long/training_large_kernel", step=5)
        graph_backtest_data(backtest_data, "cache/plots/long/backtest_large_kernel")
