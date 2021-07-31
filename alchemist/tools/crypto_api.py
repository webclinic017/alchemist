import datetime
import torch as T
import torch.nn.functional as F
from alchemist.data.crypto_data import CryptoData
from alchemist.agents.classifier_agent import ClassifierAgent

# NOTE: This currently untested, as most of it is impossible to test
# automatically. However, an effort should be made to test as much as possible

class CryptoAPI():

    def __init__(self, pairs, agent_path, n_features):
        self.pairs = pairs
        self.agent_path = agent_path
        self.n_features = n_features

    def update_holdings(self):
        # Load most recent data
        print(self.pairs)
        to_date = datetime.datetime.today()
        from_date = to_date - datetime.timedelta(days=self.n_features)
        data = CryptoData(self.pairs, from_date, to_date)
        data.format_data_into_percentages()
        # NOTE: This should be a part of CryptoData, not this messy thing
        x_data = []
        reordered_df = data.percentage_data.reorder_levels([1, 0], 1)
        for pair in self.pairs:
            relevant_df = reordered_df[pair]
            x_data.append([relevant_df.iloc[1:].values])

        # Load agent
        data.generate_datasets(n_features=self.n_features)
        agent = ClassifierAgent(data.train_ds)
        agent.load_chkpt(path=self.agent_path)

        # Make decision
        decision_list = []
        for x in x_data:
            x = T.tensor(x, device=agent.device, dtype=T.float32)
            x = T.unsqueeze(x, 1)
            prediction = agent.forward(x)
            prediction = F.softmax(prediction, dim=1)
            classes = T.argmax(prediction, dim=1)
            decision_list.append(classes)
        print(decision_list)

        decided_pairs = [p for i, p in enumerate(self.pairs) 
                         if decision_list[i][0] == 1]

        print(decided_pairs)

        # Send commands to API
        
    def sell_all():
        pass

    def buy_crypto(pairs):
        # n_pairs = len(pairs)
        symbol = "BTC"
        pass
