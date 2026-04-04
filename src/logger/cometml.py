from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd


class CometMLWriter:
    """
    Class for experiment tracking via CometML
    + local scalar logging to jsonl/csv.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        log_to_local=True,
        local_log_filename="metrics.jsonl",
        local_summary_filename="metrics_latest.csv",
        **kwargs,
    ):
        self.logger = logger
        self.project_config = project_config
        self.run_id = run_id
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

        # ---------- local logging ----------
        self.log_to_local = log_to_local
        self._local_rows = []

        save_dir = Path(project_config["trainer"]["save_dir"]) / project_config["writer"]["run_name"]
        self.local_run_dir = Path(save_dir)
        self.local_run_dir.mkdir(parents=True, exist_ok=True)

        self.local_log_path = self.local_run_dir / local_log_filename
        self.local_summary_path = self.local_run_dir / local_summary_filename

        # ---------- comet ----------
        self.exp = None
        self.comet_available = False

        try:
            import comet_ml

            comet_ml.login()

            resume = project_config["trainer"].get("resume_from") is not None

            if resume:
                if mode == "offline":
                    exp_class = comet_ml.ExistingOfflineExperiment
                else:
                    exp_class = comet_ml.ExistingExperiment

                self.exp = exp_class(experiment_key=self.run_id)
            else:
                if mode == "offline":
                    exp_class = comet_ml.OfflineExperiment
                else:
                    exp_class = comet_ml.Experiment

                self.exp = exp_class(
                    project_name=project_name,
                    workspace=workspace,
                    experiment_key=self.run_id,
                    log_code=kwargs.get("log_code", False),
                    log_graph=kwargs.get("log_graph", False),
                    auto_metric_logging=kwargs.get("auto_metric_logging", False),
                    auto_param_logging=kwargs.get("auto_param_logging", False),
                )
                self.exp.set_name(run_name)
                self.exp.log_parameters(parameters=project_config)

            self.comet_available = True
            self.comet_ml = comet_ml


    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step

        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            seconds = max(duration.total_seconds(), 1e-12)
            self.add_scalar("steps_per_sec", (self.step - previous_step) / seconds)
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{object_name}_{self.mode}"

    def _write_local_scalar(self, scalar_name, scalar):
        if not self.log_to_local:
            return

        row = {
            "timestamp": datetime.now().isoformat(),
            "step": int(self.step),
            "mode": self.mode,
            "name": self._object_name(scalar_name),
            "value": float(scalar),
        }

        with open(self.local_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._local_rows.append(row)

        try:
            latest = {}
            for r in self._local_rows:
                latest[r["name"]] = {
                    "timestamp": r["timestamp"],
                    "step": r["step"],
                    "mode": r["mode"],
                    "name": r["name"],
                    "value": r["value"],
                }
            pd.DataFrame(latest.values()).to_csv(self.local_summary_path, index=False)
        except Exception as e:
            self.logger.warning(f"Не удалось сохранить локальный CSV с метриками: {e}")

    def add_checkpoint(self, checkpoint_path, save_dir):
        if self.comet_available and self.exp is not None:
            self.exp.log_model(
                name="checkpoints",
                file_or_folder=checkpoint_path,
                overwrite=True,
            )

    def add_scalar(self, scalar_name, scalar):
        scalar = float(scalar)

        self._write_local_scalar(scalar_name, scalar)

        if self.comet_available and self.exp is not None:
            self.exp.log_metrics(
                {self._object_name(scalar_name): scalar},
                step=self.step,
            )

    def add_scalars(self, scalars):
        for scalar_name, scalar in scalars.items():
            self._write_local_scalar(scalar_name, float(scalar))

        if self.comet_available and self.exp is not None:
            self.exp.log_metrics(
                {
                    self._object_name(scalar_name): float(scalar)
                    for scalar_name, scalar in scalars.items()
                },
                step=self.step,
            )

    def add_image(self, image_name, image):
        if self.comet_available and self.exp is not None:
            self.exp.log_image(
                image_data=image,
                name=self._object_name(image_name),
                step=self.step,
            )

    def add_audio(self, audio_name, audio, sample_rate=None):
        if self.comet_available and self.exp is not None:
            audio = audio.detach().cpu().numpy().T
            self.exp.log_audio(
                file_name=self._object_name(audio_name),
                audio_data=audio,
                sample_rate=sample_rate,
                step=self.step,
            )

    def add_text(self, text_name, text):
        if self.comet_available and self.exp is not None:
            self.exp.log_text(
                text=text,
                step=self.step,
                metadata={"name": self._object_name(text_name)},
            )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        if self.comet_available and self.exp is not None:
            values_for_hist = values_for_hist.detach().cpu().numpy()
            self.exp.log_histogram_3d(
                values=values_for_hist,
                name=self._object_name(hist_name),
                step=self.step,
            )

    def add_table(self, table_name, table: pd.DataFrame):
        if self.comet_available and self.exp is not None:
            self.exp.set_step(self.step)
            self.exp.log_table(
                filename=self._object_name(table_name) + ".csv",
                tabular_data=table,
                headers=True,
            )

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()