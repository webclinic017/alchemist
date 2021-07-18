import datetime
import unittest
from alchemist.agents.new_agent import *
from alchemist.data.ticker_data import *


# Test how many workers gives optimal speed in various conditions
class TestNumWorkers(unittest.TestCase):

    def test_basic_conditions(self):
        path = "cache/tests/data/test_num_workers_basic"
        data = get_data(path = path, tickers = ["AAPL", "GOOG", "AMZN"], 
                        from_date = "2015-01-01", to_date = "2020-01-01")
        data = format_into_percentages(data, formatting_basis = "daily close")
        x, y = format_into_xy(data, num_features = 5)
        dataset = TickerDataset(x, y)

        time_list = []
        for n in range(10):
            agent = Agent(dataset, num_workers = n)
            start = datetime.datetime.now()
            agent._train(epochs = 10)
            end = datetime.datetime.now()
            time_taken = end - start
            time_list.append(str(time_taken))
            print(time_taken)

        print(time_list)


    def test_longer_data(self):
        path = "cache/tests/data/test_num_workers_long_data"
        data = get_data(path = path, tickers = ["AAPL", "GOOG", "AMZN"], 
                        from_date = "2015-01-01", to_date = "2020-01-01")
        data = format_into_percentages(data, formatting_basis = "daily close")
        x, y = format_into_xy(data, num_features = 30)
        dataset = TickerDataset(x, y)

        time_list = []
        for n in range(10):
            agent = Agent(dataset, num_workers = n)
            start = datetime.datetime.now()
            agent._train(epochs = 10)
            end = datetime.datetime.now()
            time_taken = end - start
            time_list.append(str(time_taken))
            print(time_taken)

        print(time_list)
