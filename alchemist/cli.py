import fire
import logging
from alchemist.tools.crypto_api import CryptoAPI
from alchemist.data.crypto_data import CryptoData
from alchemist.agents.classifier_agent import ClassifierAgent


# NOTE: This is currently untested, as it is hard to test effectively,
# though some way should be found to do so.

class crypto():

    def __init__(self, pairs=["BTC-USD", "ETH-USD"]):
        self.pairs = pairs
        self.agent_path = "cache/agents/cryptoAgent"
        self.n_features = 21

    def generate_agent(self, epochs=300):
        # TODO: Add more options for agent training etc.?
        logging.basicConfig(level=20)
        data = CryptoData(self.pairs, "2000-01-01", "2021-01-01", balance=True,
                          adjust_volatility=False, n_features=self.n_features,
                          divider=1.02)
        agent = ClassifierAgent(train_ds=data.train_ds)
        agent.train_(epochs=epochs)
        agent.save_chkpt(path=self.agent_path)

    def backtest(self):
        data = CryptoData(self.pairs, "2021-01-01", "2021-08-01", balance=True,
                          adjust_volatility=False, n_features=self.n_features,
                          backtest_dataset=True)
        agent = ClassifierAgent(backtest_ds=data.backtest_ds)
        agent.load_chkpt(self.agent_path)
        agent.backtest()

    def update_holdings(self):
        # Invoke API
        api = CryptoAPI(pairs=self.pairs, agent_path=self.agent_path,
                        n_features=self.n_features)
        api.update_holdings()


if __name__ == "__main__":
    fire.Fire()
