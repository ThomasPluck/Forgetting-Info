import torch
from torch import nn
from torch.nn import functional as F

class MovingKernelDeltaLayer(nn.Module):
    """
    A neural network layer implementing the Moving Kernel Delta algorithm.

    This layer updates its parameters based on moving averages of kernel operations
    on activations and weights.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output features.
        activation (callable): Activation function to be applied.
        gamma (float): Parameter for the moving average calculation.
        epsilon (float): Learning rate for gradient descent.
        eta (float): Scaling factor for the noise process.
        process (callable): Function defining the noise process.
        kernel_op (callable): Kernel operation to be applied.
        device (torch.device): Device on which to create the layer.
        noise_source (callable, optional): Function to generate noise. Defaults to torch.randn_like.

    Attributes:
        weights (nn.Parameter): Weight matrix of the layer.
        biases (nn.Parameter): Bias vector of the layer.
        Ezw, Ew, Ezb, Eb, Ez (torch.Tensor): Moving averages for various quantities.
        eta_w, eta_b (torch.Tensor): Learning rates for weights and biases.
        prev_weights, prev_biases (torch.Tensor): Previous weights and biases for kernel calculation.
    """

    def __init__(self, input_size, output_size, activation,
                 gamma, epsilon, eta, process, kernel_op,
                 device, noise_source = torch.randn_like):
        super(MovingKernelDeltaLayer, self).__init__()
        
        # Initialize parameters and hyperparameters
        self.device = device
        self.weights = nn.Parameter(torch.randn(output_size, input_size, device=device))
        self.biases = nn.Parameter(torch.randn(output_size, device=device))
        self.activation = activation
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta
        self.process = process
        self.kernel_op = kernel_op
        self.noise_source = noise_source

        # Initialize moving averages and learning rates
        self.Ezw = torch.zeros(output_size, input_size, device=device)
        self.Ew = torch.zeros(output_size, input_size, device=device)
        self.Ezb = torch.zeros(output_size, device=device)
        self.Eb = torch.zeros(output_size, device=device)
        self.Ez = torch.zeros(1, device=device)

        self.eta_w = torch.zeros(output_size, input_size, device=device)
        self.eta_b = torch.zeros(output_size, device=device)

        self.prev_weights = torch.zeros_like(self.weights)
        self.prev_biases = torch.zeros_like(self.biases)

    def forward(self, x, eval=False):
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.
            eval (bool, optional): If True, skip updating moving averages. Defaults to False.

        Returns:
            torch.Tensor: Output after applying weights, biases, and activation function.
        """
        z = F.linear(x, self.weights, self.biases)
        activated_z = self.activation(z)
        if not eval:
            self.update_moving_averages(activated_z)
            self.compute_eta()
        return activated_z
    
    def update_moving_averages(self, z):
        """
        Update the moving averages based on the current activation and weights.

        Args:
            z (torch.Tensor): Current layer activation.
        """
        weights = torch.stack((self.weights, self.prev_weights), dim=0)
        biases = torch.stack((self.biases, self.prev_biases), dim=0)
        
        delta_kz = self.kernel_op(z, axis=1)
        delta_kw = self.kernel_op(weights)
        delta_kb = self.kernel_op(biases)
        delta_kzw = delta_kz * delta_kw
        delta_kzb = delta_kz * delta_kb

        self.Ezw = (1 - self.gamma) * self.Ezw + self.gamma * torch.mean(delta_kzw,dim=0)
        self.Ezb = (1 - self.gamma) * self.Ezb + self.gamma * torch.mean(delta_kzb,dim=0)
        self.Ew = (1 - self.gamma) * self.Ew + self.gamma * torch.mean(delta_kw, dim=0)
        self.Ez = (1 - self.gamma) * self.Ez + self.gamma * torch.mean(delta_kz, dim=0)
        self.Eb = (1 - self.gamma) * self.Eb + self.gamma * torch.mean(delta_kb, dim=0)

    def compute_eta(self):
        """
        Compute the learning rates for weights and biases based on moving averages.
        """
        self.eta_w = self.eta * torch.abs(self.Ezw - self.Ez * self.Ew)
        self.eta_b = self.eta * torch.abs(self.Ezb - self.Ez * self.Eb)

    def update_parameters(self, dLdW, dLdb):
        """
        Update the layer parameters based on gradients and noise process.

        Args:
            dLdW (torch.Tensor): Gradient of the loss with respect to weights.
            dLdb (torch.Tensor): Gradient of the loss with respect to biases.
        """
        with torch.no_grad():
            self.prev_weights = self.weights
            self.prev_biases = self.biases

            self.weights += -self.epsilon * dLdW + self.eta_w * self.process(self.weights,noise_source=self.noise_source)
            self.biases += -self.epsilon * dLdb + self.eta_b * self.process(self.biases,noise_source=self.noise_source)

class KernelDeltaLayer(nn.Module):
    """
    A neural network layer implementing the Kernel Delta algorithm.

    This layer updates its parameters based on kernel operations on inputs and weights.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output features.
        activation (callable): Activation function to be applied.
        epsilon (float): Learning rate for gradient descent.
        eta (float): Scaling factor for the noise process.
        process (callable): Function defining the noise process.
        kernel_op (callable): Kernel operation to be applied.
        device (torch.device): Device on which to create the layer.
        noise_source (callable, optional): Function to generate noise. Defaults to torch.randn_like.

    Attributes:
        weights (nn.Parameter): Weight matrix of the layer.
        biases (nn.Parameter): Bias vector of the layer.
        eta_w, eta_b (torch.Tensor): Learning rates for weights and biases.
    """

    def __init__(self, input_size, output_size,
                 activation, epsilon, eta, process,
                 kernel_op, device, noise_source = torch.randn_like):
        super(KernelDeltaLayer, self).__init__()

        self.device = device
        self.weights = nn.Parameter(torch.randn(output_size, input_size, device=device))
        self.biases = nn.Parameter(torch.randn(output_size, device=device))
        self.activation = activation
        self.epsilon = epsilon
        self.eta = eta
        self.process = process
        self.kernel_op = kernel_op
        self.noise_source = noise_source

        self.eta_w = torch.zeros_like(self.weights)
        self.eta_b = torch.zeros_like(self.biases)

    def forward(self, x, train=False):
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.
            train (bool, optional): If True, compute kernel delta covariance. Defaults to False.

        Returns:
            torch.Tensor: Output after applying weights, biases, and activation function.
        """
        z = F.linear(x, self.weights, self.biases)
        activated_z = self.activation(z)
        if train:
            self.compute_kernel_delta_covariance(x)
        return activated_z

    def compute_kernel_delta_covariance(self, x):
        """
        Compute the kernel delta covariance and update learning rates.

        Args:
            x (torch.Tensor): Input tensor.
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            
            # Create perturbed weight and bias tensors
            noise_w = self.eta_w * self.noise_source(batch_size, *self.weights.shape, device=self.device)
            noise_b = self.eta_b * self.noise_source(batch_size, *self.biases.shape, device=self.device)

            perturbed_weights = self.weights.unsqueeze(0).expand(batch_size, -1, -1) + (noise_w * self.eta_w.unsqueeze(0))
            perturbed_biases = self.biases.unsqueeze(0).expand(batch_size, -1) + (noise_b * self.eta_b.unsqueeze(0))

            # Compute Ew and Ez
            Ew = self.kernel_op(perturbed_weights)
            Ex = self.kernel_op(x, dim=1)
            Eb = self.kernel_op(perturbed_biases)

            Ewx = Ew * Ex
            Ebx = Eb * Ex
            
            # Compute covariance
            cov_w = torch.mean(Ewx,dim=0) - torch.mean(Ew, dim=0) * torch.mean(Ex, dim=0)
            cov_b = torch.mean(Ebx,dim=0) - torch.mean(Eb, dim=0) * torch.mean(Ex, dim=0)
            
            # Update eta values
            self.eta_w = self.eta * cov_w
            self.eta_b = self.eta * cov_b

    def update_parameters(self, dLdW, dLdb):
        """
        Update the layer parameters based on gradients and noise process.

        Args:
            dLdW (torch.Tensor): Gradient of the loss with respect to weights.
            dLdb (torch.Tensor): Gradient of the loss with respect to biases.
        """
        with torch.no_grad():
            # Add only the deterministic component of the OU process
            process_w = self.process(self.weights,noise_source=self.noise_source)
            process_b = self.process(self.biases,noise_source=self.noise_source)
            
            self.weights += -self.epsilon * dLdW + self.eta * self.eta_w * process_w
            self.biases += -self.epsilon * dLdb + self.eta * self.eta_b * process_b