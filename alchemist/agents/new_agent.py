import os
import math
import datetime
import warnings
import torch as T
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent(nn.Module):

    def __init__(self, train_ds, test_ds = None, batch_size = 128, learning_rate = 1e-2,
                 num_workers = 0):
        super(Agent, self).__init__()

        self.n_features = len(train_ds.x_data[-1])
        self.feature_length = len(train_ds.x_data[-1][0])
        self.batch_size = batch_size
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # Layers, but not like an oinion
        kernel_size = 3
        padding = math.floor(kernel_size / 2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size, padding = padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size, padding = padding, stride = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, padding = padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size, padding = padding, stride = 1)
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
        # print(train_ds.x_data)
        self.train_data_loader = T.utils.data.DataLoader(train_ds,
                    batch_size = batch_size, shuffle=True, num_workers=num_workers)
        # for x, y in enumerate(self.train_data_loader): print(x, y)
        self.test_data_loader = None if test_ds == None else (
                T.utils.data.DataLoader(train_ds, batch_size = batch_size, 
                                        shuffle=True, num_workers=num_workers))

        # For documentation, testing etc.
        self.loss_history = []
        self.acc_history = []

    def calc_input_dims(self, n_features, feature_length):
        # NOTE: This is very necessary!!! It's the input dims of the first
        #       fully connected layer, not of the whole network!!!
        x = T.zeros((1, 1, n_features, feature_length))
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        return int(np.prod(x.size()))


    def forward(self, x):
        # Make sure input is a tensor, and push it to gpu if possible
        # This throws a warning that I can't fix, so we ignore it
        x = x.to(self.device)
        # x = x.type(T.float32)
        # Apply convolutional layers, relu and maxpool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
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

    def _train(self, epochs, verbose = False):
        # Introduces some random variation to aid with training
        self.train()
        # Clear acc and loss history for documentation
        acc_history = []
        loss_history = []

        for ep in range(epochs):
            # print(len(enumerate(train_data_loader)))
            ep_loss = 0
            ep_acc = []
            # print("-------------", self.train_data_loader.x_data)
            for j, (input, label) in enumerate(self.train_data_loader):
                # --- Very vital steps ---
                self.optimizer.zero_grad()
                # This throws a warning that I can't fix, so we ignore it
                # with warnings.catch_warnings():
                    # warnings.simplefilter("ignore")
                label = label.to(self.device, dtype = T.long)
                input = input.to(dtype = T.float32)
                # print(type(input))

                # print(input.shape)
                # input = T.tensor(input)
                prediction = self.forward(input)
                # prediction = prediction.to(dtype = T.float64)
                # prediction = T.squeeze(prediction)
                # print(prediction, label)
                # print(prediction.dtype, input.dtype)
                # exit()
                loss = self.loss(prediction, label)
                loss.backward()
                self.optimizer.step()
                # +_+_+_+_ previous
                # ---                  ---
                # +++ Pretty much just documentation +++
                # print(prediction)
                prediction = F.softmax(prediction, dim=1)
                # print(prediction)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.Tensor([1.]).to(self.device),
                                T.Tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_loss += loss.item()
                ep_acc.append(acc.item())
                # +++                                +++
            acc_history.append(np.mean(ep_acc))
            loss_history.append(ep_loss)

            # NOTE: "verbose" needs better implementation
            if verbose: print("Finished epoch", ep, 
                              " total loss % .3f" % ep_loss,
                              " accuracy %.3f" % np.mean(ep_acc))

        train_hist = pd.DataFrame({"epoch" : range(1, epochs + 1),
                                   "acc" : acc_history,
                                   "loss" : loss_history})
        return train_hist

    def _test(self, epochs = 5, verbose = False):
        # Makes sure there isn't any random variation, as in training
        self.eval()
        # Clear history for documentation
        loss_history = []
        acc_history = []

        for ep in range(epochs):
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(self.test_data_loader):
                # --- Steps not needed for testing ---
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                # ---                              ---
                # +++ The rest of it +++
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

                # NOTE: Below is code for recording things that tripped up
                #       the agent, currently inactive but supposedly functional.
                # fails = []
                # # Convert tensors to lists
                # classes = classes.tolist()
                # label = label.tolist()
                # input = input.tolist()
                # # Get indexes of wrong responses
                # for k in range(len(classes)):
                #     if classes[k] != label[k]:
                #         # Add the correct label that was guessed wrong
                #         fails.append(label[k])
                # # Only print fails if there are any
                # if len(fails) != 0:
                #     # print(fails)
                #     # Identify the actual lists that tripped us up
                #     for k in fails:
                #         bad_list = input[k]
                #         # print(bad_list)
                #         self.fails.append(bad_list)

                ep_loss += loss.item()
                ep_acc.append(acc.item())
            acc_history.append(np.mean(ep_acc))
            loss_history.append(ep_loss)
                # +++                +++

            # NOTE: "verbose" needs better implementation
            if verbose: print("Finished epoch", ep, 
                              " total loss % .3f" % ep_loss,
                              " accuracy %.3f" % np.mean(ep_acc))
            # print("Finished epoch", i, " total loss % .3f" % ep_loss,
                    # "accuracy %.3f" % np.mean(ep_acc))
        return np.mean(acc_history), np.mean(loss_history)




