from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler, Subset, ConcatDataset
from collections import defaultdict
import itertools
import random

class SingleClassSequentialSampler(Sampler):
    """
    A custom sampler that yields batches of data from a single class at a time.

    This sampler is designed to create batches where all examples are from the same class,
    cycling through all classes sequentially.

    Args:
        targets (list): The list of target labels for the dataset.
        batch_size (int): The size of each batch.

    Attributes:
        indices_by_label (dict): A dictionary mapping each label to a list of indices.
        labels (list): A sorted list of unique labels in the dataset.
        num_batches (int): The total number of batches that will be generated.
    """

    def __init__(self, targets, batch_size):
        self.targets = targets
        self.batch_size = batch_size
        self.indices_by_label = defaultdict(list)
        for idx, label in enumerate(self.targets):
            self.indices_by_label[label.item()].append(idx)
        self.labels = sorted(self.indices_by_label.keys())
        
        self.num_batches = sum(len(indices) for indices in self.indices_by_label.values()) // self.batch_size

    def __iter__(self):
        """
        Returns an iterator over the batches.

        Each batch contains indices for examples of a single class.
        """
        all_batches = []
        for label in self.labels:
            label_indices = self.indices_by_label[label]
            label_batches = [label_indices[i:i + self.batch_size] for i in range(0, len(label_indices), self.batch_size)]
            all_batches.extend(label_batches)
        
        return iter(itertools.chain.from_iterable(all_batches))

    def __len__(self):
        """
        Returns the number of batches in an epoch.
        """
        return self.num_batches

def get_mnist_train_loader(batch_size):
    """
    Creates a DataLoader for the MNIST training set with random shuffling.

    Args:
        batch_size (int): The size of each batch.

    Returns:
        DataLoader: A DataLoader for the MNIST training set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_sequential_mnist_loader(batch_size, train=True):
    """
    Creates a DataLoader for the MNIST dataset using the SingleClassSequentialSampler.

    Args:
        batch_size (int): The size of each batch.
        train (bool, optional): If True, use the training set; otherwise, use the test set. Defaults to True.

    Returns:
        DataLoader: A DataLoader for the MNIST dataset with sequential class sampling.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    sampler = SingleClassSequentialSampler(dataset.targets, batch_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def get_hardcore_sequential_mnist_loader(batch_size, repeats):
    """
    Creates a DataLoader for a 'hardcore' version of MNIST where each class is represented by a single, repeated example.

    Args:
        batch_size (int): The size of each batch.
        repeats (int): The number of times each selected example should be repeated.

    Returns:
        DataLoader: A DataLoader for the 'hardcore' MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

    # Group indices by class
    class_indices = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Select one random example from each class
    selected_indices = [random.choice(indices) for indices in class_indices]

    # Create a new dataset with repeated examples
    hardcore_datasets = []
    for idx in selected_indices:
        single_example_dataset = Subset(dataset, [idx] * repeats)
        hardcore_datasets.append(single_example_dataset)

    # Concatenate all the repeated examples
    hardcore_dataset = ConcatDataset(hardcore_datasets)

    return DataLoader(hardcore_dataset, batch_size=batch_size, shuffle=False)

def get_mnist_test_loader(batch_size):
    """
    Creates a DataLoader for the MNIST test set with random shuffling.

    Args:
        batch_size (int): The size of each batch.

    Returns:
        DataLoader: A DataLoader for the MNIST test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)