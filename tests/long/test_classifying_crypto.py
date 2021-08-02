import logging
import unittest
from tests.utils.graphing import *
from alchemist.data.crypto_data import *
from alchemist.agents.classifier_agent import *
from alchemist.tools.crypto_list import crypto_dict

# pairs = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "DOT-USD", 
         # "UNI-USD"]
pairs = [p for p in crypto_dict.keys()][-250:]

class TestMain(unittest.TestCase):

    def test_main(self):
        data = CryptoData(pairs=pairs,
                          from_date="2014-01-01", to_date="2019-01-01",
                          train_fraction=0.8, n_features=28,
                          adjust_volatility=False, divider=1, balance=True)
        data_b = CryptoData(pairs=pairs,
                            from_date="2019-01-01", to_date="2021-01-01",
                            backtest_dataset=True, n_features=28,
                            adjust_volatility=False)
        # data_b.generate_datasets(adjust_volatility=False, n_features=28,
                                 # train_fraction=5e-2)
        agent = ClassifierAgent(data.train_ds, data.test_ds, 
                                backtest_ds=data_b.backtest_ds)
        logging.basicConfig(level=15)
        train_hist = agent.train_(epochs=80)
        agent.test_()
        backtest_hist, av_e, tot_e = agent.backtest()
        graph_train_data(train_hist, 
                         path="cache/tests/plots/crypto/basicTrain")
        graph_backtest_data(backtest_hist, 
                            path="cache/tests/plots/crypto/basicBacktest")
        agent.save_chkpt(path="cache/tests/agents/basicCryptoAgent")

    def _test_backtest(self):
        data_b = CryptoData(pairs=pairs,
                            from_date="2018-01-01", to_date="2021-01-01",
                            backtest_dataset=True, n_features=28,
                            adjust_volatility=False)
        data_b.generate_datasets(adjust_volatility=False, n_features=28,
                                 train_fraction=5e-2)
        agent = ClassifierAgent(test_ds=data_b.test_ds, 
                                backtest_ds=data_b.backtest_ds)
        logging.basicConfig(level=15)
        agent.load_chkpt(path="cache/tests/agents/basicCryptoAgent")
        agent.test_()
        backtest_hist, av_e, tot_e = agent.backtest()
        graph_train_data(train_hist, 
                         path="cache/tests/plots/crypto/basicTrain")
        graph_backtest_data(backtest_hist, 
                            path="cache/tests/plots/crypto/basicBacktest")
