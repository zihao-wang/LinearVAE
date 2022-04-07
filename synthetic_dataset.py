import torch
from torch.utils.data import TensorDataset


def get_general_vae_dataset(num_samples: int,
                            input_dim: int,
                            target_dim: int,
                            singular_values: list):
    """
    preliminary implementation
    we have the parameters better off.
    """
    x = torch.randn(size=(num_samples, input_dim))
    rank = min(input_dim, target_dim, len(singular_values))

    transfer_matrix = torch.zeros(input_dim, target_dim)
    transfer_matrix[:rank, :rank] = torch.diag(
        torch.tensor(singular_values[:rank]))

    y = x.mm(transfer_matrix)

    print(transfer_matrix)
    return TensorDataset(x, y)
    

