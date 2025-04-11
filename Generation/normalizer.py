import torch

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, data):
        """
        Fit normalizer on a (N, D) tensor.
        """
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        # To avoid division by zero
        self.std[self.std == 0] = 1e-8
        self.fitted = True

    def normalize(self, data):
        """
        Normalize using stored mean and std.
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet. Call `.fit(data)` first.")
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """
        Reverts normalization.
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet. Call `.fit(data)` first.")
        return data * self.std + self.mean

    def to(self, device):
        """
        Move mean and std to a given device (e.g., GPU).
        """
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self