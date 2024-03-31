import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SubsetWithTargets(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

class SubsetWithTargetsSingleChannel(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image = torch.repeat_interleave(image, 3, 0)
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class ConcatWithTargets(Dataset):
    r"""
    Concat of a dataset at specified indices.
    """
    def __init__(self, dataset1, dataset2):
        self.dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        self.targets = torch.Tensor(list(dataset1.targets) + list(dataset2.targets)).type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class LabeledToUnlabeledDataset(Dataset):
    r"""
    Remove labels from a labeled dataset.
    """
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset

    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data

    def __len__(self):
        return len(self.wrapped_dataset)

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        if len(self.X.shape) == 2:
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.Y = torch.tensor(self.Y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]

        # Convert image back to PIL Image to apply torchvision transforms

        if self.transform:
            if len(self.X.shape) > 2:
                image = transforms.ToPILImage()(image)
            image = self.transform(image)

        # If you want the image to be a tensor again, ensure transform includes ToTensor()
        return image, label