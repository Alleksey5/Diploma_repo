# train.py (фрагмент)
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.trainer import Trainer   # НЕ GanTrainer


from src.datasets.data_utils import get_dataloaders
from src.trainer import GanTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifigan")
def main(config):
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = "cuda" if (config.trainer.device == "auto" and torch.cuda.is_available()) else config.trainer.device
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(model)

    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    epoch_len = config.trainer.get("epoch_len")
    train_loader = dataloaders["train"]
    steps_per_epoch = int(epoch_len) if epoch_len is not None else len(train_loader)
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
    epochs = int(config.trainer.n_epochs)

    # --- 2 OPTIMIZERS ---
    gen_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    disc_params = list(model.mpd.parameters()) + list(model.msd.parameters())

    gen_optimizer = instantiate(config.gen_optimizer, params=gen_params)
    disc_optimizer = instantiate(config.disc_optimizer, params=disc_params)

    # schedulers (можно пока только для G)
    gen_lr_scheduler = None
    if config.get("gen_lr_scheduler") is not None:
        gen_lr_scheduler = instantiate(config.gen_lr_scheduler, optimizer=gen_optimizer)

    disc_lr_scheduler = None
    if config.get("disc_lr_scheduler") is not None:
        disc_lr_scheduler = instantiate(config.disc_lr_scheduler, optimizer=disc_optimizer)

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),

        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        gen_lr_scheduler=gen_lr_scheduler,
        disc_lr_scheduler=disc_lr_scheduler,
    )


    trainer.train()


if __name__ == "__main__":
    main()
