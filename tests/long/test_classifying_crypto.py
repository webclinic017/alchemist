import logging
import unittest
from tests.utils.graphing import *
from alchemist.data.crypto_data import *
from alchemist.agents.classifier_agent import *

pairs = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "DOT-USD", 
         "UNI-USD"]

class TestMain(unittest.TestCase):

    def test_main(self):
        data = CryptoData(pairs=pairs,
                          from_date="2001-01-01", to_date="2018-01-01",
                          train_fraction=0.8, n_features=21,
                          adjust_volatility=False, divider=1.02, balance=True)
        data_b = CryptoData(pairs=pairs,
                            from_date="2018-01-01", to_date="2021-01-01",
                            backtest_dataset=True, n_features=21,
                            adjust_volatility=False)
        agent = ClassifierAgent(data.train_ds, data.test_ds, 
                                backtest_ds=data_b.backtest_ds)
        logging.basicConfig(level=15)
        train_hist = agent.train_(epochs=300)
        agent.test_()
        backtest_hist, av_e, tot_e = agent.backtest()
        graph_train_data(train_hist, 
                         path="cache/tests/plots/crypto/basicTrain")
        graph_backtest_data(backtest_hist, 
                            path="cache/tests/plots/crypto/basicBacktest")
        agent.save_chkpt(path="cache/tests/agents/basicCryptoAgent")

