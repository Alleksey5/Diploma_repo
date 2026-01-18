import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()


def custom_collate(batch):
    """
    Custom collate function that pads audio tensors to the closest power of two in time dimension.

    Args:
        batch (list): List of dataset items, each containing multiple segments.

    Returns:
        dict: Collated batch with padded tensors.
    """
    batch_dict = {"audio": [], "tg_audio": [], "file_id": [], "size": 0}

    for sample in batch:
        for segment in sample:  # Each sample is a list of segments
            batch_dict["audio"].append(segment["audio"])
            batch_dict["tg_audio"].append(segment["tg_audio"])
            batch_dict["file_id"].append(segment["file_id"])
            batch_dict["size"] = segment["size"]

    # Найдём максимальную длину и ближайшую степень двойки
    max_len = max([x.shape[-1] for x in batch_dict["audio"]])
    pad_len = closest_power_of_two(max_len)

    def pad_to_power_of_two(x):
        pad_size = pad_len - x.shape[-1]
        return F.pad(x, (0, pad_size), mode="constant", value=0)

    batch_dict["audio"] = torch.stack([pad_to_power_of_two(x) for x in batch_dict["audio"]])
    if all(x.shape == batch_dict["tg_audio"][0].shape for x in batch_dict["tg_audio"]):
        batch_dict["tg_audio"] = torch.stack(batch_dict["tg_audio"])
    else:
        batch_dict["tg_audio"] = batch_dict["tg_audio"]

    return batch_dict