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


def run_experiment(dataset_name: str, input_len: int, use_ght: bool, graph_mode: str, dual_stream: bool) -> None:
    output_len = 24
    batch_size = 16 if input_len >= 1440 else 64

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
        ght_graph_mode=graph_mode,
        ght_ema_decay=0.99,
        ght_ema_update_every=1,
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
        batch_size=batch_size,
        metrics=["MSE", "MAE"],
        loss="MSE",
        seed=42,
        lr=0.001,
    )

    BasicTSLauncher.launch_training(cfg)


def main():
    for dataset_name in ["ETTh1", "ETTh2"]:
        for input_len in [96, 1440]:
            print(f"\n===== {dataset_name} L={input_len}: Baseline PatchTST =====")
            run_experiment(dataset_name, input_len, use_ght=False, graph_mode="rule", dual_stream=False)

            print(f"\n===== {dataset_name} L={input_len}: GHT Rule (single) =====")
            run_experiment(dataset_name, input_len, use_ght=True, graph_mode="rule", dual_stream=False)

            print(f"\n===== {dataset_name} L={input_len}: GHT EMA (single) =====")
            run_experiment(dataset_name, input_len, use_ght=True, graph_mode="ema", dual_stream=False)

            print(f"\n===== {dataset_name} L={input_len}: GHT Dual-Stream (rule) =====")
            run_experiment(dataset_name, input_len, use_ght=True, graph_mode="rule", dual_stream=True)


if __name__ == "__main__":
    main()
