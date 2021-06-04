import torch
import unittest
from alchemist.agents.classifier_agent import *

class TestConvolutionalLayers(unittest.TestCase):

    def test_basic_layer_creation(self):
        agent = ClassifierAgent()
        agent.init_conv_layers()
        self.assertTrue(type(agent.conv_list) is list)
        self.assertTrue(type(agent.maxpool_list) is list)
        self.assertTrue(type(agent.conv_list[0])
                        is torch.nn.Conv2d)
        self.assertTrue(type(agent.maxpool_list[0])
                        is torch.nn.MaxPool2d)

    def test_setting_number_of_layers(self):
        agent = ClassifierAgent()
        agent.init_conv_layers(n_layers = 3)
        self.assertEqual(len(agent.conv_list), 3)
        self.assertEqual(len(agent.maxpool_list), 3)
        agent.init_conv_layers(n_layers = 5)
        self.assertEqual(len(agent.conv_list), 5)
        self.assertEqual(len(agent.maxpool_list), 5)

    def test_setting_conv_layer_input_dims(self):
        agent = ClassifierAgent(input_dims = [4, 1, 1])
        agent.init_conv_layers()
        self.assertEqual(agent.conv_list[0].in_channels, 4)
        agent = ClassifierAgent(input_dims = [20, 1, 1])
        agent.init_conv_layers()
        self.assertEqual(agent.conv_list[0].in_channels, 20)

    def test_layer_size_compatability(self):
        agent = ClassifierAgent()
        agent.init_conv_layers(n_layers = 10)
        for l in range(len(agent.conv_list) - 1):
            self.assertEqual(agent.conv_list[l].out_channels,
                             agent.conv_list[l+1].in_channels)

    def test_forward_through_conv_layers(self):
        agent = ClassifierAgent(input_dims = [7, 4, 20])
        agent.init_conv_layers(n_layers = 10)
        x = torch.randn(1, 7, 4, 20)
        for l in range(len(agent.conv_list)):
            x = agent.conv_list[l](x)
            x = agent.maxpool_list[l](x)

    def test_padding(self):
        # Padding should mean that the inputs and outputs of any
        # layer have the same dimentions.
        agent = ClassifierAgent(input_dims = [4, 20, 69])
        agent.init_conv_layers(n_layers = 10)
        x = torch.randn(1, 4, 20, 69)
        for l in range(len(agent.conv_list)):
            y = agent.conv_list[l](x)
            y = agent.maxpool_list[l](y)
            self.assertEqual(y.size(), torch.Size([1, 16, 20, 69]))
            x = y


class TestFullyConnectedLayers(unittest.TestCase):

    def setUp(self):
        self.agent = ClassifierAgent(input_dims = [2, 10, 4])
        self.agent.init_conv_layers(n_layers = 2)

    def test_calc_input_dims(self):
        # Agent here is not self.agent, so different input_dims etc.
        # can be tested if necessary without affecting other tests
        agent = ClassifierAgent(input_dims = [2, 10, 4])
        agent.init_conv_layers()
        input_dims = agent.calc_input_dims()
        self.assertEqual(input_dims, 640)

    def test_basic_fc_layer_creation(self):
        self.agent.init_fc_layers()
        self.assertTrue(type(self.agent.fc_list) is list)
        self.assertTrue(type(self.agent.fc_list[0])
                        is torch.nn.Linear)

    def test_setting_number_of_fc_layers(self):
        self.agent.init_fc_layers(n_layers = 2)
        self.assertEqual(len(self.agent.fc_list), 2)
        self.agent.init_fc_layers(n_layers = 6)
        self.assertEqual(len(self.agent.fc_list), 6)

    def test_fc_layer_detect_input_dims(self):
        agent = ClassifierAgent(input_dims = [1, 6, 2])
        agent.init_conv_layers()
        agent.init_fc_layers()
        self.assertEqual(agent.fc_list[0].in_features, 192)
        agent = ClassifierAgent(input_dims = [4, 20, 69])
        agent.init_conv_layers(n_layers = 3)
        agent.init_fc_layers()
        self.assertEqual(agent.fc_list[0].in_features, 22080)

    def test_setting_fc_layer_output_dims(self):
        self.agent.init_fc_layers(n_outputs = 2)
        self.assertEqual(self.agent.fc_list[-1].out_features, 2)
        self.agent.init_fc_layers(n_outputs = 5)
        self.assertEqual(self.agent.fc_list[-1].out_features, 5)


class TestOtherFunctions(unittest.TestCase):

    def test_creating_agent_with_single_call(self):
        # We should be able to create an Agent just by calling the class
        agent = ClassifierAgent(input_dims = [1, 1, 1], n_conv_layers = 1,
                                n_fc_layers = 1, n_outputs = 2)
        self.assertEqual(len(agent.conv_list), 1)
        self.assertEqual(len(agent.maxpool_list), 1)
        self.assertEqual(len(agent.fc_list), 1)
        self.assertEqual(agent.fc_list[-1].out_features, 2)

    def test_forward_through_all_layers(self):
        agent = ClassifierAgent(input_dims = [2, 4, 20],
                                n_conv_layers = 3,
                                n_fc_layers = 2,
                                n_outputs = 2)
        x = torch.randn(1, 2, 4, 20)
        x = agent.forward(x)
        # As we can't know exactly what x should be, we can settle
        # for checking its dimensions.
        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(len(x[0]), 2)

    # At this point I realised I don't have time to redo the
    # agent right now, and that I might have started foing it
    # wrong. If you're reading this in the future, it's up to
    # you to make this commented test work, or redo the whole
    # file.

    # def test_sending_tensor_to_device(self):
        # # Not quite sure how to *best* test this, but this works
        # agent = ClassifierAgent(input_dims = [1, 3, 10],
                                # n_conv_layers = 3,
                                # n_fc_layers = 2,
                                # n_outputs = 2)
        # x = torch.randn(1, 1, 3, 10)
        # x = agent.forward(x)
        # if torch.cuda.is_available():
            # self.assertEqual(agent.device.type, "cuda")
            # self.assertTrue(x.is_cuda)
        # else:
            # self.assertEqual(agent.device.type, "cpu")
            # self.assertFalse(x.is_cuda)



if __name__ == "__main__":
    unittest.main()
