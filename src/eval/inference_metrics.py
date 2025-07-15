import json
from pathlib import Path

import torch
import pandas as pd
from typeguard import typechecked
from torchmetrics import MetricCollection

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryROC,
)


class InferenceMetrics:
    columns = ["Accuracy", "Precision", "Recall"]  # , "F1Score"]

    def __init__(self, threshold: float = 0.5, multidim_average: str = "samplewise"):
        self.metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=threshold, multidim_average=multidim_average),
                BinaryPrecision(threshold=threshold, multidim_average=multidim_average),
                BinaryRecall(threshold=threshold, multidim_average=multidim_average),
            ]
        )
        self.names = []
        self.false_positives = []
        self.true_positives = []
        self.thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    @typechecked
    def update(self, scores: torch.Tensor, expected_scores: torch.Tensor, sample_path: Path | list[Path]):
        prediction = torch.clamp(scores, min=0.0, max=1.0)
        target = expected_scores.bool().long()
        roc = BinaryROC(thresholds=self.thresholds)
        if target.get_device() >= 0:
            roc.to(target.device)
        false_positives, true_positives, _ = roc(prediction, target)
        self.false_positives.append(false_positives)
        self.true_positives.append(true_positives)

        # We define a hyperparameter threshold (offset) which shifts the range to negative values
        # After that we will clamp so everything negative becomes "falsy"
        # This way we can still use metrics as for classification case

        self.metrics.update(prediction, target)
        names = [sp.name for sp in sample_path] if isinstance(sample_path, list) else [sample_path.name]
        names = [int(name.removesuffix(".pt")) for name in names]
        self.names.extend(names)

    def reset(self):
        self.metrics.reset()
        self.names = []
        self.preds = []
        self.targets = []

    def finalize(self, save_dir=None) -> pd.DataFrame:
        data = []
        for name in self.metrics.keys():
            if name.removeprefix("Binary") in InferenceMetrics.columns:
                value = self.metrics[name].compute()
                data.append(value)
        data = torch.stack(data).T
        if data.dim() == 1:
            # to avoid "shape of passed value is X, indices imply Y" error
            data = data.unsqueeze(0)

        df = pd.DataFrame(columns=InferenceMetrics.columns, data=data.tolist())
        df.insert(loc=0, column="Name", value=self.names)

        if save_dir is not None:
            false_positives = torch.stack(self.false_positives, dim=0).sum(dim=0) / len(self.false_positives)
            true_positives = torch.stack(self.true_positives, dim=0).sum(dim=0) / len(self.false_positives)
            result = {
                "false_positives": false_positives.tolist(),
                "true_positives": true_positives.tolist(),
                "thresholds": self.thresholds,
            }
            with open(save_dir / "roc.json", "w") as f:
                json.dump(result, f, indent=2)
        self.reset()

        return df
