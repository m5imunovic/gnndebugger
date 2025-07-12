import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from data.lja_to_graph import lja_to_graph
from utils.cfg_helpers import instantiate_component_list
from utils.logger import get_logger
from utils.path_helpers import get_config_root

log = get_logger(__name__)
torch.multiprocessing.set_start_method("spawn", force=True)


def partition_results(results: list[torch.Tensor], dataset_info_path) -> list[dict]:
    """Partition results from trainer.predict() into separate lists.

    Args:
        results (List[torch.Tensor]): results from trainer.predict()

    Returns:
        List[Dict]: partitioned results
    """

    mapper = {0: "dropped", 1: "kept"}
    partitions = []
    for idx, result in enumerate(results):
        selection = {"dropped": [], "kept": []}
        id_map_path = dataset_info_path / f"{idx}.idmap"
        with open(id_map_path) as handle:
            id_map = json.load(handle)
        assert len(result) == len(id_map), "Inference result and corresponding id map do not match."
        for i, val in enumerate(result.int()):
            hash_id = id_map[str(i)]
            key = mapper[val.item()]
            selection[key].append(hash_id)

        partitions.append(selection)

    return partitions


def infere(cfg: DictConfig) -> None:
    """Inference using the pretrained model on given dataset.

    Args:
        cfg (DictConfig): Configurations for inference
    """
    if cfg.paths.data_dir is None:
        log.error("Paths data dir is not defined. Aborting!")
        return
    model_path = cfg.model_path or None
    if model_path is None or not Path(model_path).exists():
        log.error(f"No model path {model_path} provided.")
        return

    if cfg.lja_mode:
        log.info("Converting LJA graph...")
        lja_to_graph(Path(cfg.datamodules.dataset_path), create_test_dir=True)
    log.info("Init data module...")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodules)
    log.info("Init model...")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.models)

    # count and print the number of parameters
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total number of parameters: {total_params}")
    log.info("Init loggers...")
    loggers = instantiate_component_list(cfg.loggers)

    log.info("Init trainer/inference module...")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    # log.info("Save experiment hyperparameters...")
    # TODO: implement saving

    log.info("Start inference...")
    if cfg.lja_mode:
        predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=model_path)
        log.info("Prediction finished.")
        return predictions
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=model_path)

    log.info("Experiment finished.")
    return


@hydra.main(
    version_base=None,
    config_path=str(get_config_root()),
    config_name="inference_cfg.yaml",
)
def main(cfg: DictConfig) -> None:
    """Inference script entry point.

    Args:
        cfg (DictConfig): composed configuration

    Returns:
        Optional[float]: output of training function
    """
    if "only_print_config" in cfg:
        print(OmegaConf.to_yaml(cfg))
        return

    return infere(cfg)


if __name__ == "__main__":
    main()
