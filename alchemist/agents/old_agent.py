import os
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings

from matplotlib import pyplot as plt

from data import TickerDataset

class Net(nn.Module):
    def __init__(self, parameters, update_data = False, batch_size = 128,
                 train_needed = True, test_needed = True, model_num = 0,
                 fname = "cache/models/trained_model"):
        super(Net, self).__init__()
        self.fname = fname
        self.model_num = model_num
        self.params = parameters
        self.batch_size = batch_size
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # NOTE: Minimum len_input is 11
        len_input = parameters["len_input"]
        self.len_input = len_input
        # TODO: This code supposedly adapts the size of the NN to the len_input,
        # this should be checked

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv1d(16, 32, 3 + math.ceil((len_input-12)/3))
        self.maxpool2 = nn.MaxPool1d(2 + math.ceil((len_input-12)/7))
        # Need to calculate the input dims for the fully connected layers
        self.input_dims = self.calc_input_dims(len_input)
        # The Fully connected layers
        self.fc1 = nn.Linear(self.input_dims, self.input_dims * 2)
        self.fc2 = nn.Linear(self.input_dims * 2, self.input_dims)
        self.fc3 = nn.Linear(self.input_dims, 2)

        self.optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

        # The training data
        self.get_data(parameters = parameters, update_data = update_data,
                      train_needed = train_needed, test_needed = test_needed)

        # For documentation purposes
        self.loss_history = []
        self.acc_history = []
        self.fails = []

    def reset(self):
        # NOTE: Currently just re-generates all the layers.
        # There may be a better way to do this
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv1d(16, 32, 3 + math.ceil((self.len_input-12)/3))
        self.maxpool2 = nn.MaxPool1d(2 + math.ceil((self.len_input-12)/7))
        # Need to calculate the input dims for the fully connected layers
        self.input_dims = self.calc_input_dims(self.len_input)
        # The Fully connected layers
        self.fc1 = nn.Linear(self.input_dims, self.input_dims * 2)
        self.fc2 = nn.Linear(self.input_dims * 2, self.input_dims)
        self.fc3 = nn.Linear(self.input_dims, 2)

        self.optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

    def calc_input_dims(self, len_input):
        # NOTE: This is very necessary!!! It's the input dims of the first
        #       fully connected layer, not of the whole network!!!
        x = T.zeros((1, 1, len_input, 4))
        x = self.conv1(x)
        x = self.maxpool1(x)
        # Squeeze the input from 2 to 1 dimentions
        x = x.squeeze(3)
        x = self.conv2(x)
        x = self.maxpool2(x)

        return int(np.prod(x.size()))

    def get_data(self, parameters, train_needed = True, test_needed = True,
                 update_data = False):
        if train_needed:
            print("Loading train data ...")
            # Ticker data is retrieved using the TickerDataset class
            ticker_train_data = TickerDataset(parameters = parameters,
                                              train = True,
                                              update_data = update_data)
            # Then it is formatted with PyTorch's DataLoader
            self.train_data_loader = T.utils.data.DataLoader(ticker_train_data,
                                                batch_size = self.batch_size,
                                                shuffle=True,
                                                num_workers=8)
        if test_needed:
            print("Loading test data ...")
            # Test data is a random set of data not used for training
            ticker_test_data = TickerDataset(parameters = parameters,
                                             train = False,
                                             update_data = update_data)
            self.test_data_loader = T.utils.data.DataLoader(ticker_test_data,
                                                batch_size = self.batch_size,
                                                shuffle=True,
                                                num_workers=8)


    def forward(self, x):
        # Make sure input is a tensor, and push it to gpu if possible
        # This throws a warning that I can't fix, so we ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = T.tensor(x).to(self.device)
        x = x.type(T.float32)
        # Apply convolutional layers, relu and maxpool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        # Squeeze the input from 2 to 1 dimentions
        x = x.squeeze(3)
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

    def _train(self, epochs):
        # Introduces some random variation to aid with training
        self.train()
        # Clear acc and loss history for documentation
        self.acc_history = []
        self.loss_history = []

        for i in range(epochs):
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(self.train_data_loader):
                # --- Very vital steps ---
                self.optimizer.zero_grad()
                # This throws a warning that I can't fix, so we ignore it
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    label = T.tensor(label, dtype = T.long).to(self.device)
                prediction = self.forward(input)
                # print(prediction.dtype, input.dtype)
                # exit()
                loss = self.loss(prediction, label)
                loss.backward()
                self.optimizer.step()
                # ---                  ---
                # +++ Pretty much just documentation +++
                prediction = F.softmax(prediction, dim=1)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.Tensor([1.]).to(self.device),
                                T.Tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_loss += loss.item()
                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
            self.loss_history.append(ep_loss)
                # +++                                +++

            print("Finished epoch", i, " total loss % .3f" % ep_loss,
                  "accuracy %.3f" % np.mean(ep_acc))

    def _test(self, epochs):
        # Makes sure there isn't any random variation, as in training
        self.eval()
        # Clear history for documentation
        self.fails = []
        self.loss_history = []
        self.acc_history = []

        for i in range(epochs):
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(self.test_data_loader):
                # --- Steps not needed for testing ---
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                # ---                              ---
                # +++ The rest of it +++
                # This throws a warning that I can't fix, so we ignore it
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    label = T.tensor(label, dtype = T.long).to(self.device)
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
                self.acc_history.append(acc.item())
            self.loss_history.append(ep_loss)
                # +++                +++

            print("Finished epoch", i, " total loss % .3f" % ep_loss,
                    "accuracy %.3f" % np.mean(ep_acc))


    def save_chkpt(self, fname = None):
        # Compile the fname based on parameters
        if fname == None:
            fname = self.fname
        old_fname = fname
        for i in self.params:
            fname += "-"
            fname += str(self.params[i])
        fname += "-" + str(self.model_num)
        # Make sure that the folders the file is meant to be in exist
        path = fname[:fname.rindex("/")]
        if not os.path.exists(path):
            print("Path", path, "does not exist, creating necessary folders...")
            os.makedirs(path)
        # Then save the file
        state = {
            "number" : self.model_num,
            "state_dict" : self.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        T.save(state, fname)

    def load_chkpt(self, fname = None):
        # Compile the fname based on parameters
        if fname == None:
            fname = self.fname
        old_fname = fname
        for i in self.params:
            fname += "-"
            fname += str(self.params[i])
        fname += "-" + str(self.model_num)
        # Load the checkpoint
        state = T.load(fname)
        # Initialize state_dict from checkpoint to model
        self.load_state_dict(state['state_dict'])
        # Initialize optimizer from checkpoint to optimizer
        self.optimizer.load_state_dict(state['optimizer'])


if __name__ == "__main__":
    parameters = {"target_rise" : 0.025, "len_input" : 120}
    print("Initializing Network")
    network = Net(parameters = parameters, batch_size=128, update_data = False)
    print("Training Netork")
    network._train(epochs=100)
    print("Plotting results")
    plt.plot(network.loss_history)
    plt.savefig("cache/plots/Training Loss History")
    plt.clf()
    plt.plot(network.acc_history)
    plt.savefig("cache/plots/Training Accuracy History")
    print("Training Network")
    network._test(epochs=10)
    print("Plotting results")
    plt.clf()
    plt.plot(network.loss_history)
    plt.savefig("cache/plots/Testing Loss History")
    plt.clf()
    plt.plot(network.acc_history)
    plt.savefig("cache/plots/Testing Accuracy History")
