import torch.nn as nn
import torch.nn.functional as F


class ScalarMLP(nn.Module):
    def __init__(self, num_features, input_indices, activation=nn.Tanh(), num_hidden_layers=1, layer_widths=[4], last_activation = None, device='cpu'):
        """Initialize a SimpleNN model.

        Args:
            num_features (int): The number of input features.
            activation (torch.nn.Module, optional): The activation function to use. Defaults to nn.ReLU().
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 1.
            layer_widths (list, optional): The width of each hidden layer. Defaults to [4].

        Raises:
            AssertionError: If the length of layer_widths is not equal to num_hidden_layers.
        """
        super(ScalarMLP, self).__init__()
        self.num_features = num_features
        self.input_indices = input_indices
        assert len(layer_widths) == num_hidden_layers, "Length of layer_widths must be equal to num_hidden_layers"
        self.device = device
        layers = []
        layers.append(nn.Linear(num_features, layer_widths[0]))
        layers.append(activation)
        
        for i in range(1, num_hidden_layers):
            layers.append(nn.Linear(layer_widths[i-1], layer_widths[i]))
            layers.append(activation)
        
        layers.append(nn.Linear(layer_widths[-1], 1))
        
        self.layers = nn.Sequential(*layers)
        self.last_activation = last_activation

    def forward(self, x):
        out = self.layers(x)
        if self.last_activation is not None :
            return self.last_activation(out)
        return out
