import torch
from torch.utils.data import TensorDataset

def get_synthetic_vae_dataset(num_samples: int,
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

def get_general_vae_dataset(num_samples: int,
                            input_dim: int,
                            target_dim: int,
                            singular_values: list,):
    """
    preliminary implementation
    we have the parameters better off.
    """
    sigma = torch.tensor(singular_values).reshape(1, -1)
    _x = torch.randn(size=(num_samples, input_dim))
    x = _x * sigma

    S = torch.tensor([1.2, 1.4, 1.6, 1.8, 2], dtype=torch.float)
    oldtransform = torch.randn(input_dim, target_dim)
    U, _, Vh = torch.linalg.svd(oldtransform)
    transform = U @ torch.diag(S) @ Vh

    sigma = sigma.squeeze()
    y = x.mm(transform)
    Z = y.T.mm(_x) / len(_x)
    _, newS, _ = torch.linalg.svd(Z)
    print(newS)

    return TensorDataset(x, y)
