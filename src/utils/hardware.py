import torch

def device() -> torch.device:
    """
    Returns the device to be used for computation.

    Returns:
        torch.device: device to be used for computation
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')