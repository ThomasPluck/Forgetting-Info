import torch
import tqdm

def train_and_evaluate_network(network, train_loader, test_loader, epochs, loss_function, device):
    """
    Train and evaluate a neural network over multiple epochs.

    Args:
        network (nn.Module): The neural network to train and evaluate.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        epochs (int): Number of epochs to train.
        loss_function (callable): The loss function to use.
        device (torch.device): The device to run the computations on.

    Returns:
        nn.Module: The trained neural network.

    This function trains the network for the specified number of epochs, evaluating
    its performance on both the training and test sets after each epoch.
    """
    optimizer = torch.optim.SGD(network.parameters(), lr=network.epsilon)

    for epoch in range(epochs):
        # Training phase
        network.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = loss_function(output, target)
            loss.backward()
            for layer in network.layers:
                layer.update_parameters(layer.weights.grad, layer.biases.grad)

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        # Testing phase
        network.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Epoch {epoch+1} Testing"):
                data, target = data.to(device), target.to(device)
                output = network(data, eval=True)
                test_loss += loss_function(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        test_loss /= len(test_loader)
        test_accuracy = 100. * test_correct / test_total

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print("------")

    return network

def train_epoch(network, train_loader, loss_function, device):
    """
    Train the network for one epoch.

    Args:
        network (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for the training data.
        loss_function (callable): The loss function to use.
        device (torch.device): The device to run the computations on.

    Returns:
        tuple: A tuple containing:
            - float: The average training loss for the epoch.
            - float: The training accuracy (in percentage) for the epoch.

    This function trains the network for one complete pass through the training data.
    """
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    train_loss, train_correct, train_total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        for layer in network.layers:
            layer.update_parameters(layer.weights.grad, layer.biases.grad)
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        train_correct += pred.eq(target).sum().item()
        train_total += target.size(0)
    return train_loss / len(train_loader), 100. * train_correct / train_total

def evaluate(network, test_loader, loss_function, device):
    """
    Evaluate the network on a test dataset.

    Args:
        network (nn.Module): The neural network to evaluate.
        test_loader (DataLoader): DataLoader for the test data.
        loss_function (callable): The loss function to use.
        device (torch.device): The device to run the computations on.

    Returns:
        tuple: A tuple containing:
            - float: The average test loss.
            - float: The test accuracy (in percentage).

    This function evaluates the network's performance on the test dataset.
    """
    network.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
    return test_loss / len(test_loader), 100. * test_correct / test_total