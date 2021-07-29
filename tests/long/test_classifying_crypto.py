import unittest
from tests.utils.graphing import *
from alchemist.data.crypto_data import *
from alchemist.agents.new_agent import *

class TestMain(unittest.TestCase):

    def test_main(self):
        data = CryptoData(pairs=["BTC-USD", "ETH-USD"],
                          from_date="2001-01-01", to_date="2021-06-06",
                          train_fraction=0.8, n_features=10,
                          adjust_volatility=False, divider=1, balance=True)
        print(data.train_ds)
        agent = Agent(data.train_ds, data.test_ds)
        train_hist = agent._train(verbose=True, epochs=500)
        graph_train_data(train_hist, path = "cache/plots/long/basic_test")

