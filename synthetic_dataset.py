import torch
from torch.utils.data import TensorDataset


def get_general_vae_dataset(num_samples: int,
                            input_dim: int,
                            # target_dim: int,
                            singular_values: list):
    """
    preliminary implementation
    we have the parameters better off.
    """
    sigma = torch.tensor(singular_values).reshape(1, -1)
    x = torch.randn(size=(num_samples, input_dim))
    x = x * sigma

    return TensorDataset(x, x)
    

