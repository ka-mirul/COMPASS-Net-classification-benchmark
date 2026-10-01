# SAR ship classification benchmark

This repository contains the experiments used to compare SARATR-X and SAR-JEPA on NASTaR, FUSAR-Ship, and OpenSARShip. Run the experiments from `sar_ship_benchmark.ipynb`.

The datasets and model weights are not included. The files in `benchmark_outputs/` contain the results, configurations, data splits, training histories, predictions, logs, and confusion matrices. Fine-tuned checkpoints and TensorBoard files are excluded.

## Download the model code

These are the versions used for the experiments:

| Model | Repository | Commit |
|---|---|---|
| SARATR-X | [waterdisappear/SARATR-X](https://github.com/waterdisappear/SARATR-X) | `8f110555e32e5777034c5ab6729d23bcf9741936` |
| SAR-JEPA | [waterdisappear/SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA) | `1f17f9007481c563653066c0c49dbd55c0846298` |

Clone both repositories into this folder:

```bash
git clone https://github.com/waterdisappear/SARATR-X.git
git -C SARATR-X checkout 8f110555e32e5777034c5ab6729d23bcf9741936

git clone https://github.com/waterdisappear/SAR-JEPA.git
git -C SAR-JEPA checkout 1f17f9007481c563653066c0c49dbd55c0846298
```

## Download the pretrained weights

### SARATR-X

Download `checkpoint-200.pth` from the [official SARATR-X page on Hugging Face](https://huggingface.co/waterdisappear/SARATR-X/blob/main/weight/186K_all/checkpoint-200.pth). You may need to sign in and accept the access conditions first.

Save it here:

```text
SARATR-X/checkpoints/checkpoint-200.pth
```

SHA-256:

```text
aa57a46488d637a8d735c5d470b9a28b83b8e660ae98caf991cd183ae3643897
```

### SAR-JEPA

The SAR-JEPA checkpoint is available from the authors' [Kaggle model page](https://www.kaggle.com/models/liweijie19/sar-jepa). Use model version `liweijie19/sar-jepa/pytorch/default/1` and download:

```text
weights/SAR-JEPA/checkpoint-200.pth
```

Save it here:

```text
SAR-JEPA/checkpoints/weights/SAR-JEPA/checkpoint-200.pth
```

To download it with `kagglehub`:

```python
import kagglehub

kagglehub.model_download(
    "liweijie19/sar-jepa/pytorch/default/1",
    path="weights/SAR-JEPA/checkpoint-200.pth",
    output_dir="SAR-JEPA/checkpoints",
)
```

SHA-256:

```text
0894f1472f5beaad0e7663cb4da6261a9d7547e73016bf845b0b2c227d4b7ea0
```

Check the downloaded files before training:

```bash
sha256sum SARATR-X/checkpoints/checkpoint-200.pth
sha256sum SAR-JEPA/checkpoints/weights/SAR-JEPA/checkpoint-200.pth
```

## Add the datasets

Extract the datasets into these folders:

```text
NovaSAR_dataset/
FUSAR_Ship1.0/
OpenSARShip_1/
OpenSARShip_2/
```

There is no need to move or rename the images inside the datasets.

## Run an experiment

Open `sar_ship_benchmark.ipynb` and change the four settings in the first cell:

- `MODEL`: `saratrx` or `sarjepa`
- `RUN_SIZE`: `smoke` or `full`
- `TRAINING_MODE`: `linear_probe`, `partial_finetune`, or `full_finetune`
- `GOAL`: `opensar4`, `opensar6`, `nastar`, or `fusar`

A full experiment runs five fixed seeds. The notebook saves each run separately and reports the final result as mean ± standard deviation.

All training modes start from the downloaded pretrained weights. `linear_probe` trains only the classification head, `partial_finetune` also updates the last part of the encoder, and `full_finetune` updates the complete model.
