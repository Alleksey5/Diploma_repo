
import torch
import torchaudio
from tqdm.auto import tqdm
import soundfile as sf
import numpy as np
import librosa

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.metrics.rtf import RTF
from src.metrics.thop import THOPMetric


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
            self,
            model,
            config,
            device,
            dataloaders,
            save_path,
            metrics=None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
                skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition
        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and save predictions to disk.

        - Moves batch to device.
        - Processes multiple small segments and merges them back into full audio.
        - Saves predictions as audio files.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        x_segments = batch["audio"]  # (batch_size, 1, segment_size)
        wq = batch["tg_audio"]
        print(f"Input batch shape: {wq.shape}")

        outputs = self.model(x_segments)
        print(f"Model output shape: {outputs.shape}")

        if not isinstance(outputs, dict):
            outputs = {"pred_audio": outputs}

        batch.update(outputs)
        # Update RTF separately
        for met in self.metrics["inference"]:
            if isinstance(met, RTF):
                try:
                    met.update(self.model, batch["audio"])
                except Exception:
                    continue
            if isinstance(met, THOPMetric):
                met.update(self.model, batch["audio"])

        if metrics is not None:
            for met in self.metrics["inference"]:
                print(batch["pred_audio"].shape[0])
                for i in range(batch["pred_audio"].shape[0]):
                    source = batch["tg_audio"][i].squeeze().cpu().numpy()
                    predict = batch["pred_audio"][i].squeeze().cpu().numpy()
                    source = librosa.util.normalize(source[:min(len(source), len(predict))])
                    predict = librosa.util.normalize(predict[:min(len(source), len(predict))])

                    source = torch.from_numpy(source)[None, None]
                    predict = torch.from_numpy(predict)[None, None]

                    try:
                        metric_value = met(source, predict)
                        metrics.update(met.name, metric_value)
                    except Exception as e:
                        continue

        batch_size = batch["pred_audio"].shape[0]

        audio_dict = {}
        marg_dict = {}
        for i in range(batch_size):
            file_id = batch["file_id"][i]
            logits = batch["pred_audio"][i].clone().cpu().numpy()

            if file_id not in audio_dict:
                audio_dict[file_id] = []
            audio_dict[file_id].append(logits)
            marg_dict[file_id] = batch["size"]

        for file_id, segments in audio_dict.items():
            merged_audio = []

            for i, segment in enumerate(segments):
                
                if i == 0:
                    merged_audio.append(segment)
                elif i != (len(segments)-1):
                    merged_audio.append(segment[:,-self.cfg_trainer.window:])
                else:
                    merged_audio.append(segment[:, -marg_dict[file_id]:])
                    print(-marg_dict[file_id])

            full_audio = np.concatenate(merged_audio, axis=-1)
            full_audio_tensor = torch.FloatTensor(full_audio)

            if full_audio_tensor.dim() == 1:
                full_audio_tensor = full_audio_tensor.unsqueeze(0)

            torchaudio.save(
                self.save_path / part / f"output_{file_id}.wav",
                full_audio_tensor,
                16000,
                channels_first=True,
            )

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()

