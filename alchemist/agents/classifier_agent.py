import os
import math
import logging
import datetime
import warnings
import torch as T
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ClassifierAgent(nn.Module):

    def __init__(self, train_ds=None, test_ds=None, backtest_ds=None, 
                 batch_size=30, learning_rate=1e-3, num_workers=0,
                 kernel_size=3):
        super(ClassifierAgent, self).__init__()

        self.n_features = len(train_ds.x_data[0]) if train_ds != None else (
                len(backtest_ds.x_data[0][0]))
        self.feature_length = len(train_ds.x_data[0][0]) if (
                train_ds != None) else len(backtest_ds.x_data[0][0][0])
        self.batch_size = batch_size
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # Layers, but not like an oinion
        padding = math.floor(kernel_size / 2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size, padding = padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size, padding = padding, stride = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, padding = padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size, padding = padding, stride = 1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size, padding = padding)
        self.maxpool3 = nn.MaxPool2d(kernel_size, padding = padding, stride = 1)
        # Boring Layers
        self.input_dims = self.calc_input_dims(self.n_features, self.feature_length)
        self.fc1 = nn.Linear(self.input_dims, self.input_dims * 2)
        self.fc2 = nn.Linear(self.input_dims * 2, self.input_dims)
        self.fc3 = nn.Linear(self.input_dims, 2)

        # Optimizer etc.
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

        # Data loaders
        self.train_data_loader = None if train_ds == None else (
                T.utils.data.DataLoader(train_ds, batch_size = batch_size, 
                                        shuffle=True, num_workers=num_workers))
        self.test_data_loader = None if test_ds == None else (
                T.utils.data.DataLoader(test_ds, batch_size = batch_size, 
                                        shuffle=True, num_workers=num_workers))
        self.backtest_ds = backtest_ds

    def calc_input_dims(self, n_features, feature_length):
        # Calculate the input dimensions of the first fc layer
        x = T.zeros((1, 1, n_features, feature_length))
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)

        return int(np.prod(x.size()))

    def forward(self, x):
        # Make sure input is a tensor, and push it to gpu if possible
        x = x.to(self.device)
        # Apply convolutional layers, relu and maxpool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        # Rearrange input to be a 1-dimentional tensor for the fc layers
        x = x.view(x.size()[0], -1)
        # Now apply the fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def save_chkpt(self, path):
        # Make sure that the folders the file is meant to be in exist
        if not os.path.exists(path[:path.rindex("/")]):
            os.makedirs(path[:path.rindex("/")])
        # Then save the file
        state = {
            "state_dict" : self.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        T.save(state, path)

    def load_chkpt(self, path = None):
        # Load the checkpoint
        state = T.load(path)
        # Initialize state_dict from checkpoint to model
        self.load_state_dict(state['state_dict'])
        # Initialize optimizer from checkpoint to optimizer
        self.optimizer.load_state_dict(state['optimizer'])

    def train_(self, epochs):
        # Introduces some random variation to aid with training
        self.train()
        # logger, acc and loss history for documentation
        acc_history = []
        loss_history = []
        logger = logging.getLogger()

        for ep in range(epochs):
            ep_loss = 0
            ep_acc = []
            ep_start_time = datetime.datetime.now()
            for j, (input, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device, dtype = T.long)
                input = input.to(dtype = T.float32)
                prediction = self.forward(input)
                loss = self.loss(prediction, label)
                loss.backward()
                self.optimizer.step()
                
                # Record accuracy and loss
                prediction = F.softmax(prediction, dim=1)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.Tensor([1.]).to(self.device),
                                T.Tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                ep_loss += loss.item()
                ep_acc.append(acc.item())
            acc_history.append(np.mean(ep_acc))
            loss_history.append(ep_loss)
            ep_time = (datetime.datetime.now()-ep_start_time).microseconds/1e6
            logger.info("Finished epoch %s in %.3f seconds, total "
                        "loss %.3f, accuracy %.3f", ep, ep_time, ep_loss,
                        np.mean(ep_acc))
            if ep % 20 == 0:
                self.test_()
                self.train()

        train_hist = pd.DataFrame({"epoch" : range(1, epochs + 1),
                                   "acc" : acc_history,
                                   "loss" : loss_history})
        return train_hist

    def test_(self):
        # Makes sure there isn't any random variation, as in training
        self.eval()

        test_loss = 0
        test_acc = []
        for j, (input, label) in enumerate(self.test_data_loader):
            label = label.to(self.device, dtype = T.long)
            input = input.to(dtype = T.float32)
            prediction = self.forward(input)
            loss = self.loss(prediction, label)
            prediction = F.softmax(prediction, dim=1)
            classes = T.argmax(prediction, dim=1)
            wrong = T.where(classes != label,
                            T.Tensor([1.]).to(self.device),
                            T.Tensor([0.]).to(self.device))
            acc = 1 - T.sum(wrong) / self.batch_size
            test_loss += loss.item()
            test_acc.append(acc.item())
            # TODO: May be beneficial to record where mistakes where made?
        logger = logging.getLogger()
        logger.log(19, "Finished test with loss %.3f and accuracy %.3f", 
                   test_loss, np.mean(test_acc))

        return np.mean(test_acc), test_loss

    def backtest(self):
        self.eval()
        earnings_list = []

        # print(pd.DataFrame(self.backtest_ds.y_data))
        # print(pd.DataFrame(self.backtest_ds.x_data))

        # Present the test input data to the agent one day at a time
        for i in range(self.backtest_ds.length):
            x = self.backtest_ds.x_data[i]
            y = self.backtest_ds.y_data[i]
            for ah in x:
                for aa in ah:
                    for ar in aa:
                        for yh in y:
                            if ar == yh:
                                print("AHHHHH GOD WHYYYYYY")
            # print(x)
            # print(y)

            x = T.tensor(x, device=self.device, dtype=T.float32)
            x = T.unsqueeze(x, 1)
            prediction = self.forward(x)
            prediction = F.softmax(prediction, dim=1)
            classes = T.argmax(prediction, dim=1)
            # print(classes)
            earnings = [y_ for j, y_ in enumerate(y) if classes[j] == 1]
            # print(earnings)
            if len(earnings) == 0: earnings = [1]
            earnings = np.mean(earnings)
            earnings_list.append(earnings)

        print(earnings_list)
        average_earnings = np.mean(earnings_list)
        total_earnings = np.product(earnings_list)
        logger = logging.getLogger()
        logger.info("Finished backtest with average earnings %.5f and total"
                    " earnings %.3f", average_earnings, total_earnings)

        return earnings_list, average_earnings, total_earnings
