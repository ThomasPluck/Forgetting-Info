from layers import *
import torch.nn as nn

class MovingKernelDeltaNetwork(nn.Module):
    """
    A neural network composed of Moving Kernel Delta Layers.

    This network dynamically constructs layers based on the provided sizes and activations.

    Args:
        input_size (int): The size of the input to the network.
        layer_sizes (list of int): The sizes of the hidden layers.
        output_size (int): The size of the output from the network.
        activations (list of callable): Activation functions for each layer.
        gamma (float): The gamma parameter for the Moving Kernel Delta Layers.
        epsilon (float): The epsilon parameter for the Moving Kernel Delta Layers.
        eta (float): The eta parameter for the Moving Kernel Delta Layers.
        process (callable): The process function for the Moving Kernel Delta Layers.
        kernel_op (callable): The kernel operation for the Moving Kernel Delta Layers.
        device (torch.device): The device on which to create the layers.
        noise_source (callable, optional): Function to generate noise. Defaults to torch.randn_like.

    Attributes:
        layers (nn.ModuleList): The list of Moving Kernel Delta Layers in the network.
    """

    def __init__(self, input_size, layer_sizes, output_size, activations, gamma, epsilon, eta, process, kernel_op, device, noise_source=torch.randn_like):
        super(MovingKernelDeltaNetwork, self).__init__()
        self.layers = nn.ModuleList()

        for idx, layer_size in enumerate(layer_sizes):
            if idx == 0:
                self.layers.append(MovingKernelDeltaLayer(input_size, layer_size, activations[idx], gamma, epsilon, eta, process, kernel_op, device))
            elif idx == len(layer_sizes) - 1:
                self.layers.append(MovingKernelDeltaLayer(layer_sizes[idx-1], output_size, activations[idx], gamma, epsilon, eta, process, kernel_op, device))
            else:
                self.layers.append(MovingKernelDeltaLayer(layer_sizes[idx-1], layer_size, activations[idx], gamma, epsilon, eta, process, kernel_op, device))

    def forward(self, x, eval=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.
            eval (bool, optional): Whether the network is in evaluation mode. Defaults to False.

        Returns:
            torch.Tensor: The output of the network.
        """
        for layer in self.layers:
            x = layer(x, eval=eval)
        return x

class KernelDeltaNetwork(nn.Module):
    """
    A neural network composed of Kernel Delta Layers.

    This network dynamically constructs layers based on the provided sizes and activations.

    Args:
        input_size (int): The size of the input to the network.
        layer_sizes (list of int): The sizes of the hidden layers.
        output_size (int): The size of the output from the network.
        activations (list of callable): Activation functions for each layer.
        epsilon (float): The epsilon parameter for the Kernel Delta Layers.
        eta (float): The eta parameter for the Kernel Delta Layers.
        process (callable): The process function for the Kernel Delta Layers.
        kernel_op (callable): The kernel operation for the Kernel Delta Layers.
        device (torch.device): The device on which to create the layers.
        noise_source (callable, optional): Function to generate noise. Defaults to torch.randn_like.

    Attributes:
        layers (nn.ModuleList): The list of Kernel Delta Layers in the network.
    """

    def __init__(self, input_size, layer_sizes, output_size, activations, epsilon, eta, process, kernel_op, device, noise_source=torch.randn_like):
        super(KernelDeltaNetwork, self).__init__()
        self.layers = nn.ModuleList()

        for idx, layer_size in enumerate(layer_sizes):
            if idx == 0:
                self.layers.append(MovingKernelDeltaLayer(input_size, layer_size, activations[idx], epsilon, eta, process, kernel_op, device, noise_source=noise_source))
            elif idx == len(layer_sizes) - 1:
                self.layers.append(MovingKernelDeltaLayer(layer_sizes[idx-1], output_size, activations[idx], epsilon, eta, process, kernel_op, device, noise_source=noise_source))
            else:
                self.layers.append(MovingKernelDeltaLayer(layer_sizes[idx-1], layer_size, activations[idx], epsilon, eta, process, kernel_op, device, noise_source=noise_source))

    def forward(self, x, train=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.
            train (bool, optional): Whether the network is in training mode. Defaults to False.

        Returns:
            torch.Tensor: The output of the network.
        """
        for layer in self.layers:
            x = layer(x, train=train)
        return x