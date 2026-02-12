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


def run(cfg: BasicTSForecastingConfig, tag: str) -> None:
    print(f"\n===== {tag} =====")
    BasicTSLauncher.launch_training(cfg)


def main():
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"]
    dataset_filter = os.environ.get("DATASET", "all")
    if dataset_filter != "all":
        datasets = [d for d in datasets if d == dataset_filter]

    input_len = 96
    horizons = [96, 192, 336, 720]
    epochs = int(os.environ.get("EPOCHS", "1"))
    num_features = 7

    for dataset_name in datasets:
        for output_len in horizons:
            # DLinear
            run(BasicTSForecastingConfig(
                model=DLinear,
                model_config=DLinearConfig(input_len=input_len, output_len=output_len, num_features=num_features),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=False,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | DLinear | Lout={output_len}")

            # PatchTST
            run(BasicTSForecastingConfig(
                model=PatchTSTForForecasting,
                model_config=PatchTSTConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                    intermediate_size=128,
                ),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=False,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | PatchTST | Lout={output_len}")

            # iTransformer
            run(BasicTSForecastingConfig(
                model=iTransformerForForecasting,
                model_config=iTransformerConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                    hidden_size=64,
                    intermediate_size=128,
                    n_heads=1,
                    num_layers=1,
                    dropout=0.1,
                    use_revin=True,
                ),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=False,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | iTransformer | Lout={output_len}")

            # Autoformer
            run(BasicTSForecastingConfig(
                model=Autoformer,
                model_config=AutoformerConfig(
                    input_len=input_len,
                    output_len=output_len,
                    label_len=output_len // 2,
                    num_features=num_features,
                    hidden_size=128,
                    intermediate_size=256,
                    n_heads=4,
                    num_encoder_layers=2,
                    num_decoder_layers=1,
                    dropout=0.1,
                    use_timestamps=True,
                    timestamp_sizes=[24, 7, 31, 366],
                ),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=True,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | Autoformer | Lout={output_len}")

            # TimesNet
            run(BasicTSForecastingConfig(
                model=TimesNetForForecasting,
                model_config=TimesNetConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                    hidden_size=64,
                    intermediate_size=128,
                    num_layers=1,
                    dropout=0.1,
                    use_timestamps=True,
                    timestamp_sizes=[24, 7, 31, 366],
                ),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=True,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | TimesNet | Lout={output_len}")

            # TimeMixer
            run(BasicTSForecastingConfig(
                model=TimeMixerForForecasting,
                model_config=TimeMixerConfig(
                    input_len=input_len,
                    output_len=output_len,
                    num_features=num_features,
                    hidden_size=64,
                    intermediate_size=128,
                    num_layers=1,
                    dropout=0.1,
                    use_revin=True,
                    use_timestamps=True,
                    timestamp_sizes=[24, 7, 31, 366],
                ),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=True,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | TimeMixer | Lout={output_len}")

            # NLinear (FEDformer not in repo; placeholder for now)
            run(BasicTSForecastingConfig(
                model=NLinear,
                model_config=NLinearConfig(
                    input_len=input_len,
                    output_len=output_len,
                ),
                dataset_name=dataset_name,
                gpus=None,
                num_epochs=epochs,
                input_len=input_len,
                output_len=output_len,
                batch_size=16,
                use_timestamps=False,
                metrics=["MSE", "MAE"],
                loss="MSE",
                seed=42,
                lr=1e-3,
            ), f"{dataset_name} | NLinear | Lout={output_len}")


if __name__ == "__main__":
    main()
