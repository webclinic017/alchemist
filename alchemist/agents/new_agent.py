import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Agent(nn.Module):

    def __init__(self, train_ds, test_ds = None, batch_size = 128):
        super(Agent, self).__init__()

        self.n_features = len(train_ds.x_data[-1])
        self.feature_length = len(train_ds.x_data[-1][0])
        self.batch_size = batch_size
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # Layers, but not like an oinion
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.maxpool1 = nn.MaxPool2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.maxpool2 = nn.MaxPool2d(1)
        # Boring Layers
        self.input_dims = self.calc_input_dims(self.n_features, self.feature_length)
        self.fc1 = nn.Linear(self.input_dims, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 1)

        # Optimizer etc.
        self.optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

        # Data loaders
        self.train_data_loader = T.utils.data.DataLoader(train_ds,
                    batch_size = batch_size, shuffle=True, num_workers=8)
        self.test_data_loader = None if test_ds == None else (
                T.utils.data.DataLoader(train_ds, batch_size = batch_size, 
                                        shuffle=True, num_workers=8))

    def calc_input_dims(self, n_features, feature_length):
        # NOTE: This is very necessary!!! It's the input dims of the first
        #       fully connected layer, not of the whole network!!!
        x = T.zeros((1, 1, n_features, feature_length))
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        return int(np.prod(x.size()))



