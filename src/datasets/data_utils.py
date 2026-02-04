from itertools import repeat
import importlib

from hydra.utils import instantiate

from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if batch_transforms is None:
        return

    for transform_type, transforms in batch_transforms.items():
        if transforms is None:
            continue

        for transform_name, transform in transforms.items():
            if isinstance(transform, torch.nn.Module):
                transforms[transform_name] = transform.to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each dataset partition.
    Also initializes instance and batch transforms.

    Args:
        config (DictConfig): Hydra experiment config.
        device (str): Device to use for batch transforms.

    Returns:
        dataloaders (dict[DataLoader]): Dictionary containing dataloaders.
        batch_transforms (dict[Callable] | None): Batch-level transformations.
    """

    # ---------- batch_transforms: optional ----------
    batch_transforms = None
    if hasattr(config, "transforms") and config.transforms is not None:
        if hasattr(config.transforms, "batch_transforms") and config.transforms.batch_transforms is not None:
            batch_transforms = instantiate(config.transforms.batch_transforms)
            move_batch_transforms_to_device(batch_transforms, device)
    # -----------------------------------------------

    # Dynamically load collate_fn from string path in config
    collate_fn = None
    if "collate_fn" in config.dataloader and isinstance(config.dataloader.collate_fn, str):
        module_name, function_name = config.dataloader.collate_fn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        collate_fn = getattr(module, function_name)

    # Initialize dataloaders
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = instantiate(config.datasets[dataset_partition])  # Instantiate dataset

        assert config.dataloader.batch_size <= len(dataset), (
            f"Batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than dataset size ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
            collate_fn=collate_fn,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
