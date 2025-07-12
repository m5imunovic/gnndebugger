import json
import resource
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError
from torch.profiler import ProfilerActivity, record_function

from eval.inference_metrics import InferenceMetrics
from models.loss.mixture_loss import MixtureLoss
from utils.container import Container
from utils.logger import get_logger


log = get_logger(__name__)


def print_resource_memory_usage(stage):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.info(f"{stage} resource max rss is: {rss/1024} MB")


def print_proc_memory_usage(stage):
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {"VmSize": 0, "VmRSS": 0, "VmPeak": 0, "VmHWM": 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open("/proc/self/status")
        for line in status:
            for key in result:
                if key in line:
                    parts = line.strip().split(":")
                    result[key] = parts[1].strip()
                else:
                    continue
                    # print(key)
    finally:
        if status is not None:
            status.close()
    result_con = ", ".join([f"{k}: {v}" for k, v in result.items()])
    log.info(f"{stage} /proc/self/status poll: {result_con}")
    return result_con


def print_cuda_memory_alloc(stage):
    if torch.cuda.is_available():
        # torch.cuda.reset_peak_memory_stats(device=None)

        log.info(f"{stage} cuda max memory allocated: {torch.cuda.max_memory_allocated(device=None) / 1024**2}")
        log.info(f"{stage} cuda memory allocated: {torch.cuda.memory_allocated(device=None) / 1024**2}")
        log.info(f"{stage} cuda memory reserved: {torch.cuda.memory_reserved(device=None) / 1024**2}")


def postprocess_scores(data, scores, high=None, low=None, mincov=None):
    if mincov is not None:
        print("Filtering false positives")
        maybe_fp_mask = torch.logical_and(scores < 0.95, data.edge_attr[:, 0].unsqueeze(-1) < mincov)
        # print(f"{maybe_fp_mask.shape}, {maybe_fp_mask.shape}")
        scores[maybe_fp_mask] = 0.0
        print("After coverage filtering")
        print(f"Corrections: {maybe_fp_mask.squeeze(-1).nonzero().numel()}")
        print(f"{scores.shape=}")
    if high is not None and low is not None:
        print("Multiedge Corrections")
        mult_mask = data.edge_attr[:, -1].unsqueeze(-1)
        print(f"{mult_mask.shape=}")
        maybe_fn_mask = torch.logical_and(scores >= low, scores <= high)
        print(f"{maybe_fn_mask.shape=}")
        maybe_mult_fn_mask = torch.logical_and(mult_mask, maybe_fn_mask)
        print(f"Numel: {maybe_mult_fn_mask.squeeze(-1).nonzero().numel()}")
        cov_gt3_mask = data.edge_attr[:, 0].unsqueeze(-1) > mincov
        maybe_mult_fn_mask = torch.logical_and(maybe_mult_fn_mask, cov_gt3_mask)
        print(f"Numel: {maybe_mult_fn_mask.squeeze(-1).nonzero().numel()}")
        scores[maybe_mult_fn_mask] = 1.0

    return scores


class DBGRegressionModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: torch.nn.modules.loss._Loss,
        batch_size: int = 1,
        threshold: int = 0.5,  # in regression setting we use this to adapt range for softmax operation
        storage_path: Path | None = None,
        profiler: bool = False,
        postprocess: bool = True,
        high: float = None,
        low: float = None,
        mincov: float = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net
        self.postprocess = postprocess
        self.high = high
        self.low = low
        self.mincov = mincov
        self.batch_size = batch_size

        # training metrics
        self.train_metric = MeanSquaredError()
        # validation metrics
        self.val_metric = MeanSquaredError()

        # test metrics
        self.test_metrics = InferenceMetrics(threshold=self.hparams.threshold)
        self.storage_path = None
        self.profiler = profiler

        if storage_path is not None:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(exist_ok=True, parents=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def common_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.data.x
        edge_index = batch.data.edge_index
        device = edge_index.device
        edge_attr = getattr(batch.data, "edge_attr", None)
        graph_attr = getattr(batch.data, "graph_attr", None)
        log.info(f"Shape of edge_index is: {edge_attr.shape}")
        ei_ptr = torch.tensor(batch.ei_ptr, device=device)

        scores = self.net(x=x.float(), edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr, ei_ptr=ei_ptr)
        # scores = torch.clamp(scores, min=0)

        if getattr(batch.data, "y") is not None:
            y = batch.data.y
            expected_scores = y.float().unsqueeze(-1)
        else:
            expected_scores = None
        if self.postprocess:
            scores = postprocess_scores(batch.data, scores, high=self.high, low=self.low, mincov=self.mincov)
        return scores, expected_scores

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)

        if self.uses_mixture_loss():
            loss = self.hparams.criterion(batch.data.edge_index, scores, expected_scores)
        else:
            weights = None
            if getattr(batch.data, "weights", None) is not None:
                weights = batch.data.weights.unsqueeze(-1)
            if weights is not None:
                loss = self.hparams.criterion(scores, expected_scores, weights)
            else:
                loss = self.hparams.criterion(scores, expected_scores)
        self.train_metric(scores, expected_scores.int())

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        output = self.train_metric.compute()
        self.log(
            "train/mse",
            output,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        if self.uses_mixture_loss():
            loss = self.hparams.criterion(batch.data.edge_index, scores, expected_scores)
        else:
            weights = None
            if getattr(batch.data, "weights", None) is not None:
                weights = batch.data.weights.unsqueeze(-1)
            if weights is not None:
                loss = self.hparams.criterion(scores, expected_scores, weights)
            else:
                loss = self.hparams.criterion(scores, expected_scores)
        self.val_metric.update(scores, expected_scores.int())

        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_validation_epoch_end(self):
        output = self.val_metric.compute()
        self.log(
            "val/mse",
            output,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.val_metric.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        print_cuda_memory_alloc("Initial")
        print_proc_memory_usage("Initial")
        print_resource_memory_usage("Initial")

        if self.profiler:
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                with record_function("inference"):
                    scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
            log.info(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=100))
        else:
            scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)

        print_cuda_memory_alloc("Final")
        print_proc_memory_usage("Final")
        print_resource_memory_usage("Final")
        scores = scores.reshape((1, -1))
        expected_scores = expected_scores.reshape((1, -1))
        if self.storage_path is not None and self.batch_size == 1:
            torch.save(scores.cpu(), f"{self.storage_path/batch.path[0].stem}.pt")
            np.save(f"{self.storage_path/batch.path[0].stem}", scores.cpu().numpy())
            np.save(f"{self.storage_path}/expected_{batch.path[0].stem}", expected_scores.cpu().numpy())
            torch.save(batch.data, f"{self.storage_path}/transformed_{batch.path[0].stem}.pt")

        self.test_metrics.update(scores, expected_scores, batch.path)

    def on_test_end(self):
        df = self.test_metrics.finalize(self.storage_path)
        print(df.to_string())
        if hasattr(self.logger, "log_table"):
            self.logger.log_table("test/metrics", dataframe=df)

        if self.storage_path is not None and self.batch_size == 1:
            outdir = self.storage_path / "stats"
            outdir.mkdir(exist_ok=True, parents=True)
            with open(outdir / "stats.csv", "w") as f:
                df = df.sort_values(by=["Name"], ascending=True)
                df.to_csv(f, index=False)

            cfg = {
                "postprocess": self.postprocess,
                "mincov": self.mincov,
            }
            with open(outdir / "cfg.json", "w") as f:
                json.dump(cfg, f)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print_cuda_memory_alloc("Initial")
        print_proc_memory_usage("Initial")
        print_resource_memory_usage("Initial")

        if self.profiler:
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                with record_function("inference"):
                    scores, _ = self.common_step(batch, batch_idx, dataloader_idx)
            log.info(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=100))
        else:
            scores, _ = self.common_step(batch, batch_idx, dataloader_idx)

        print_cuda_memory_alloc("Final")
        print_proc_memory_usage("Final")
        print_resource_memory_usage("Final")
        scores = scores - self.hparams.threshold
        scores = torch.clamp(scores, min=0)
        if self.storage_path is not None:
            if self.batch_size == 1:
                torch.save(scores.cpu(), f"{self.storage_path/batch.path[0].stem}.pt")
                torch.save(batch.data, f"{self.storage_path}/transformed_{batch.path[0].stem}.pt")
            container = Container({"multiplicity": scores.cpu().to(torch.float32)})
            container.save(self.storage_path)
        return scores

    def uses_mixture_loss(self):
        return isinstance(self.hparams.criterion, MixtureLoss)
