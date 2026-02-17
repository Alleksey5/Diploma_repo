# src/trainer/gan_trainer.py
import torch
from src.trainer.trainer import Trainer


class GanTrainer(Trainer):
    """
    Работает с вашим Trainer.process_batch() (где используется self.gen_optimizer/self.disc_optimizer)
    и чинит сохранение/резюм для двух оптимизаторов и двух шедулеров.
    """

    def __init__(
        self,
        gen_optimizer,
        disc_optimizer,
        gen_lr_scheduler=None,
        disc_lr_scheduler=None,
        *args, **kwargs
    ):
        # ВАЖНО: не передаём optimizer=..., потому что ваш Trainer.__init__ это не принимает
        super().__init__(*args, **kwargs)

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        # чтобы BaseTrainer-логика (лог lr, чекпоинтинг, etc.) не ломалась:
        self.optimizer = self.gen_optimizer
        self.lr_scheduler = self.gen_lr_scheduler

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),

            "gen_optimizer": self.gen_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),

            "gen_lr_scheduler": self.gen_lr_scheduler.state_dict() if self.gen_lr_scheduler else None,
            "disc_lr_scheduler": self.disc_lr_scheduler.state_dict() if self.disc_lr_scheduler else None,

            "monitor_best": getattr(self, "mnt_best", None),
            "config": getattr(self, "config", None),
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        if checkpoint.get("monitor_best") is not None:
            self.mnt_best = checkpoint["monitor_best"]

        self.model.load_state_dict(checkpoint["state_dict"])

        if checkpoint.get("gen_optimizer") is not None:
            self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        if checkpoint.get("disc_optimizer") is not None:
            self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])

        if checkpoint.get("gen_lr_scheduler") is not None and self.gen_lr_scheduler is not None:
            self.gen_lr_scheduler.load_state_dict(checkpoint["gen_lr_scheduler"])
        if checkpoint.get("disc_lr_scheduler") is not None and self.disc_lr_scheduler is not None:
            self.disc_lr_scheduler.load_state_dict(checkpoint["disc_lr_scheduler"])

        # синхронизируем "общие" поля BaseTrainer
        self.optimizer = self.gen_optimizer
        self.lr_scheduler = self.gen_lr_scheduler

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
