# GnnDebugger

## Installation and development

We recommend the usage of mambaforge for setting up Python environment:

https://github.com/conda-forge/miniforge#mambaforge

```Bash
### Install conda
mamba init

### Creates new environment (named fenv)
mamba env create --file=environment.yaml
mamba activate fenv
```

### Setup pre-commit hooks

```bash
pip install pre-commit
# Set up on the first usage
pre-commit install
pre-commit run --all-files
```

This is a recommended option to work with source repository in case you want to make modifications to the codebase and configurations.

## Configuration

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. `Hydra` enables dynamic configurations and fosters modular development
of code. It is not straightforward as using standard `YAML` configuration but the learning curve definitely pays off. If you run into issues,
don't hesitate to ask for help with your setup.

### Quick Hydra Onboarding

Our top-level configuration contains following sections:

- hydra
- loggers
- paths
- callbacks
- datamodules
- models
- trainer
- experiment

`hydra` contain the logging configurations. `hydra` takes care of system logs.
`paths` configuration defines the paths used by the framework (e.g. where the datasets are located `paths.dataset_dir`, where the output files will be saved `paths.storage_dir`, the root directory of model for inference `paths.model_dir`, etc.). In the default configuration, this paths are relative to the `paths.data_dir` which is set to `null` so user needs to override it, depending on the local setup. For example, if you place your files in the following structure:

```Bash
/data
├── datasets
├── models
├── storage
...
```

you will need to override `data_dir` value like this:

```Bash
python3 src/train.py paths.data_dir=/data
```

Of course, one can override configuration options individually. E.g. if datasets is placed in the root, i.e. `/datasets` path you will run the following command:

```Bash
python3 src/train.py paths.data_dir=/data paths.dataset_dir=/datasets
```

A bit more drastic approach is to write a new config package, e.g. "paths_local.yaml" with configuration options corresponding to your setup and override the entire default package:

```Bash
python3 src/train.py paths=paths_local.yaml
```

For a temporary modifications, one can always edit the `.yaml` files but this beats the purpose of using `Hydra`.

The configuration system is tightly coupled with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/). The packages `loggers`
and `callbacks` represent a list of callback options that are being passed into `pytorch_lightning.Trainer` as arguments. `Hydra` can instantiate object from configuration using `__target__` definition (see `Hydra` official documentation for more details). The `Trainer` class itself is configured using the `training` package. E.g. if you want to change
default device (CPU) for training, you can simply specify the `accelerator` [argument](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer) as command line parameter:

```Bash
python3 src/train.py trainer.accelerator=gpu
```

The models contain the package configuring the models and optimizers used in the process.
For example, default `models.yaml` file contains the configuration for `models.dbg_light_module.DBGLightningModule`. This class accepts `net` argument where user can configure the neural network architecture model. The "module" class will submit following data arguments to the networks `forward` function (`x`- node attributes, `edge_attr` - edge attributes, `graph_attr` - graph attributes and `edge_index`). The class configured by `net` package should inherit from `torch.nn.Module`. Of course, one can implement the
entirely new lightning module and network implementation and use it instead.
The `datamodules` package configures datasets and data transformations.

### Training

It is recommended to use a machine with GPU. It might be necessary to change network configuration to adapt the memory consumption to
your local setup. For full-blown training with the released datasets, we recommend at least 40GB graphics card. To run the training with default config (`train_cfg.yaml`)
simply run:

```Bash
python3 src/train.py <override options>
```

It is recommended to define `model_output_path` with a directory path. Here, the best
model will be saved to a file `best_model.ckpt`. Otherwise, the model is also stored using the checkpoint tracking callback in the training log directories (see `model_ckpt.yaml` config package).

If you want to use other configuration, e.g. `/configurations/custom_train.yaml` located in some other directory (i.e. not in project's `config` directory) run the following command:

```Bash
python3 src/train.py --config-path /configurations --config-name custom_train.yaml
```

### Inference

To run inference, it is important to specify the path to the model and dataset name
which needs to contain `test` subfolder which contains `raw` subfolder with graph (.pt) files. If you want to save output (instead of just printing stats) define the `models.net.storage_path` config

```Bash
python3 src/inference.py model_path=<path to model> dataset_name=<name of dataset> models.net.storage_path=<output dirpath>
```

## Training (if you are not interested in development)

For re-training with already available architectures and datasets, we recommend setting up singularity image, script assume that the project is located in
`~/work/gnndebugger`

```Bash
cd apptainer && bash build_apptainer.sh
```

Configs should be modified according to the local paths (check `config/paths/paths.yaml`)
Default data directory is `/data`.
Datasets are expected in `/data/datasets` directory.
Place configuration in `/data/configs` directory.

This will build image `dbgc.sif` which can then be used for training.

```Bash
apptainer run \
        --bind $HOME/data:/data \
        --nv $HOME/work/gnndebugger/dbgc.sif \
        --config-path /data/config \
        --config-name "train.yaml" \
```

Datasets for training and tests are available at [Zenodo](https://doi.org/10.5281/zenodo.15073168), along with the sample configurations.

### Inference on the test set

We provide a small dataset in `data/datasets`, model and configuration for testing purposes. In order to run the inference first export the `PROJECT_ROOT`:

```Bash
export PROJECT_ROOT=$(pwd)
```

in the root directory of the project. The configuration system depends on this variable (see `paths.root_dir` config value).
Note that, thanks to the `Hydra` one can also override this value via command line using `paths.root_dir=<path_to_root_dir>`.

In order to run the inference with default settings simply run (with activated virtual environment):

```Bash
python3 src/inference.py --config-dir=configs --config-name=inference_test_dataset.yaml
```

By default inference is run on CPU. To switch the computation to CUDA GPU use:

```Bash
python3 src/inference.py --config-dir=configs --config-name=inference_test_dataset.yaml trainer.accelerator=cuda
```

One can metadata (stats, transformed dataset) by setting the `models.storage_path` config value to non-null value, e.g.:

```Bash
python3 src/inference.py --config-dir=configs --config-name=inference_test_dataset.yaml models.storage_path=/tmp/inference_metadata
```
