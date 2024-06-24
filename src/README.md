# Kernel Delta Networks

This repository contains an implementation of Kernel Delta Networks, a novel approach to neural network training that incorporates kernel methods and stochastic processes. The project includes custom layer implementations, network architectures, and utilities for training and evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Components](#components)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Key Features](#key-features)
6. [Examples](#examples)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Kernel Delta Networks combine traditional neural network architectures with kernel methods and stochastic processes to create a unique learning paradigm. This approach aims to improve network performance and generalization by incorporating dynamic parameter updates based on kernel operations and noise processes.

## Components

The repository consists of several key components:

1. **Custom Layers**: 
   - `MovingKernelDeltaLayer`: A layer that updates its parameters based on moving averages of kernel operations.
   - `KernelDeltaLayer`: A layer that updates its parameters based on kernel operations on inputs and weights.

2. **Network Architectures**:
   - `MovingKernelDeltaNetwork`: A network composed of Moving Kernel Delta Layers.
   - `KernelDeltaNetwork`: A network composed of Kernel Delta Layers.

3. **Kernel Operators**: Various kernel gram matrix operations, including Gaussian, cosine, dot product, L1, L2, and Lp kernels.

4. **Data Loaders**: Custom data loaders for MNIST, including sequential and "hardcore" versions.

5. **Training and Evaluation**: Functions for training and evaluating networks.

6. **Stochastic Processes**: Implementations of various noise processes, including Wiener and Ornstein-Uhlenbeck processes.

## Installation

To use this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/kernel-delta-networks.git
cd kernel-delta-networks
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to create and train a Kernel Delta Network:

```python
import torch
from networks import KernelDeltaNetwork
from training import train_and_evaluate_network
from data_loaders import get_mnist_train_loader, get_mnist_test_loader
from kernel_operators import gaussian_kernel_gram_off_diagonal
from stochastic_processes import unit_OU_process

# Define network parameters
input_size = 784  # for MNIST
layer_sizes = [128, 64]
output_size = 10
activations = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Softmax(dim=1)]

# Create the network
network = KernelDeltaNetwork(
    input_size, layer_sizes, output_size, activations,
    epsilon=0.01, eta=0.1, process=unit_OU_process,
    kernel_op=gaussian_kernel_gram_off_diagonal,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Get data loaders
train_loader = get_mnist_train_loader(batch_size=64)
test_loader = get_mnist_test_loader(batch_size=64)

# Train and evaluate the network
trained_network = train_and_evaluate_network(
    network, train_loader, test_loader,
    epochs=10, loss_function=torch.nn.CrossEntropyLoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

## Key Features

- Custom layer implementations with kernel-based parameter updates
- Flexible network architectures supporting various kernel operations
- Integration of stochastic processes for noise generation
- Custom data loaders for MNIST with different sampling strategies
- Comprehensive training and evaluation utilities

## Examples

The repository includes several example scripts demonstrating how to use different components:

- `example_moving_kernel_delta.py`: Demonstrates the use of Moving Kernel Delta Networks
- `example_kernel_delta.py`: Shows how to create and train a Kernel Delta Network
- `example_custom_mnist_loader.py`: Illustrates the use of custom MNIST data loaders

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.