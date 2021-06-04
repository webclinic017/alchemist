import math
import torch
import torch.nn as nn
import numpy as np

class ClassifierAgent(nn.Module):

    def __init__(self, input_dims = [1, 1, 1], n_conv_layers = None,
                 n_fc_layers = None, n_outputs = None):
        super(ClassifierAgent, self).__init__()
        self.input_dims = input_dims
        # self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   # else "cpu")
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2,
                                         # momentum=0.9)
        # self.loss = nn.CrossEntropyLoss()
        # This conditional creation of essential parts of the network
        # may have no use apart from in testing, and may even be bad
        if n_conv_layers != None:
            self.init_conv_layers(n_layers = n_conv_layers)
        if n_fc_layers != None:
            self.init_fc_layers(n_layers = n_fc_layers,
                                n_outputs = n_outputs)


    def init_conv_layers(self, n_layers = 1, in_channels = 1,
                         channels = 16, kernel_size = 3):
        self.conv_list = []
        self.maxpool_list = []
        in_channels = self.input_dims[0]
        # Padding means that no matter how many layers there are,
        # the input size stays the same. Maybe unhealthy.
        # The most obvious problem with this is that it results in
        # very large fc layers, which may be bad for performance
        padding = math.floor(kernel_size / 2)
        for l in range(n_layers):
            self.conv_list.append(nn.Conv2d(in_channels = in_channels,
                                            out_channels = channels,
                                            kernel_size = kernel_size,
                                            padding = padding))
            in_channels = channels # Kinda inefficient
            # Changes also made to maxpool to maintain size of data,
            # also maybe a bad idea long-term
            self.maxpool_list.append(nn.MaxPool2d(kernel_size = kernel_size,
                                                  padding = padding,
                                                  stride = 1))

    def calc_input_dims(self):
        x = torch.randn(1, self.input_dims[0],
                        self.input_dims[1],
                        self.input_dims[2])
        for l in range(len(self.conv_list)):
            x = self.conv_list[l](x)
            x = self.maxpool_list[l](x)

        return int(np.prod(x.size()))

    def init_fc_layers(self, n_layers = 1, n_outputs = 1):
        self.fc_list = []
        n_inputs = self.calc_input_dims()
        self.fc_list = []
        for l in range(n_layers - 1):
            # The way n_out_temp is made can be varied, it's just
            # there to make the number of neurons go down, for
            # the computer's sake.
            n_out_temp = math.ceil((n_inputs - n_outputs) * 2/3)
            self.fc_list.append(nn.Linear(n_inputs, n_out_temp))
            n_inputs = n_out_temp
        self.fc_list.append(nn.Linear(n_inputs, n_outputs))
        # Fc layers are last to be created, so now we push
        # everything to gpu if needed
        # self.to(self.device)

    def forward(self, x):
        # Send tensor to gpu for speeed
        # x = x.to(self.device)
        # Forward through conv layers
        for l in range(len(self.conv_list)):
            x = self.conv_list[l](x)
            x = self.maxpool_list[l](x)
        # Rearrange to fit into fc layers
        x = x.view(x.size()[0], -1)
        # Forward through fc layers
        for layer in self.fc_list:
            x = layer(x)

        return x


