import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.PatchTST import (
    PatchTSTConfig,
    PatchTSTForForecasting,
    PatchTSTForForecastingGHT,
)


def run_experiment(dataset_name: str, use_ght: bool, dual_stream: bool) -> None:
    input_len = 96
    output_len = 24

    model_cfg = PatchTSTConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
        intermediate_size=128,
        use_ght=use_ght,
        ght_ratio=0.5,
        ght_k=8,
        ght_assign_hidden=128,
        ght_use_gnn=True,
        ght_local_window=None,
        ght_temperature=1.0,
        ght_dual_stream=dual_stream,
        ght_ratio_trend=0.125,
        ght_ratio_residual=0.5,
        ght_fuse="concat",
    )

    model_cls = PatchTSTForForecastingGHT if use_ght else PatchTSTForForecasting

    cfg = BasicTSForecastingConfig(
        model=model_cls,
        model_config=model_cfg,
        dataset_name=dataset_name,
        gpus=None,
        num_epochs=1,
        input_len=input_len,
        output_len=output_len,
        batch_size=16,
        metrics=["MSE", "MAE"],
        loss="MSE",
        seed=42,
        lr=0.001,
    )

    BasicTSLauncher.launch_training(cfg)


def main():
    for dataset_name in ["ETTh1", "ETTh2"]:
        print(f"\n===== {dataset_name}: Baseline PatchTST =====")
        run_experiment(dataset_name, use_ght=False, dual_stream=False)
        print(f"\n===== {dataset_name}: PatchTST + GHT Dual-Stream =====")
        run_experiment(dataset_name, use_ght=True, dual_stream=True)


if __name__ == "__main__":
    main()
