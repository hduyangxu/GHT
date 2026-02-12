import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts.models.Autoformer import Autoformer, AutoformerConfig
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.TimeMixer import TimeMixerConfig, TimeMixerForForecasting
from basicts.models.NLinear import NLinear, NLinearConfig
from basicts.runners.callback import EarlyStopping


def run(cfg: BasicTSForecastingConfig, tag: str) -> None:
    print(f"\n===== {tag} =====")
    BasicTSLauncher.launch_training(cfg)


def build_common_cfg(dataset_name: str, input_len: int, output_len: int, use_timestamps: bool) -> dict:
    return dict(
        dataset_name=dataset_name,
        gpus="0",
        num_epochs=30,
        input_len=input_len,
        output_len=output_len,
        batch_size=64,
        use_timestamps=use_timestamps,
        train_data_num_workers=2,
        train_data_pin_memory=True,
        val_data_num_workers=2,
        val_data_pin_memory=True,
        test_data_num_workers=2,
        test_data_pin_memory=True,
        metrics=["MSE", "MAE"],
        target_metric="MSE",
        best_metric="min",
        loss="MSE",
        callbacks=[EarlyStopping(patience=5)],
        seed=42,
        lr=1e-3,
    )


def main():
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"]
    horizons = [96, 192, 336, 720]
    input_len = 96
    num_features = 7

    for dataset_name in datasets:
        for output_len in horizons:
            label_len = input_len // 2

            # DLinear
            run(BasicTSForecastingConfig(
                model=DLinear,
                model_config=DLinearConfig(input_len=input_len, output_len=output_len, num_features=num_features),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=False),
            ), f"{dataset_name} | DLinear | Lout={output_len}")

            # PatchTST
            run(BasicTSForecastingConfig(
                model=PatchTSTForForecasting,
                model_config=PatchTSTConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                ),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=False),
            ), f"{dataset_name} | PatchTST | Lout={output_len}")

            # iTransformer
            run(BasicTSForecastingConfig(
                model=iTransformerForForecasting,
                model_config=iTransformerConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                ),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=False),
            ), f"{dataset_name} | iTransformer | Lout={output_len}")

            # Autoformer (requires timestamps)
            run(BasicTSForecastingConfig(
                model=Autoformer,
                model_config=AutoformerConfig(
                    input_len=input_len,
                    output_len=output_len,
                    label_len=label_len,
                    num_features=num_features,
                    use_timestamps=True,
                    timestamp_sizes=[24, 7, 31, 366],
                ),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=True),
            ), f"{dataset_name} | Autoformer | Lout={output_len}")

            # TimesNet (requires timestamps)
            run(BasicTSForecastingConfig(
                model=TimesNetForForecasting,
                model_config=TimesNetConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                    use_timestamps=True,
                    timestamp_sizes=[24, 7, 31, 366],
                ),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=True),
            ), f"{dataset_name} | TimesNet | Lout={output_len}")

            # TimeMixer (requires timestamps)
            run(BasicTSForecastingConfig(
                model=TimeMixerForForecasting,
                model_config=TimeMixerConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                    use_timestamps=True,
                    timestamp_sizes=[24, 7, 31, 366],
                ),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=True),
            ), f"{dataset_name} | TimeMixer | Lout={output_len}")

            # NLinear (FEDformer not available in repo)
            run(BasicTSForecastingConfig(
                model=NLinear,
                model_config=NLinearConfig(
                    input_len=input_len,
                    output_len=output_len,
                ),
                **build_common_cfg(dataset_name, input_len, output_len, use_timestamps=False),
            ), f"{dataset_name} | NLinear | Lout={output_len}")


if __name__ == "__main__":
    main()
