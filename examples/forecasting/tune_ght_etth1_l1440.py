import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting, PatchTSTForForecastingGHT


def run_experiment(model_cls, model_cfg, dataset_name: str, input_len: int, epochs: int) -> None:
    output_len = 24
    batch_size = 16
    cfg = BasicTSForecastingConfig(
        model=model_cls,
        model_config=model_cfg,
        dataset_name=dataset_name,
        gpus=None,
        num_epochs=epochs,
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
    dataset_name = "ETTh1"
    input_len = 1440
    epochs = 3

    print(f"\n===== {dataset_name} L={input_len}: Baseline PatchTST ({epochs} epochs) =====")
    baseline_cfg = PatchTSTConfig(
        input_len=input_len,
        output_len=24,
        num_features=7,
        intermediate_size=128,
        use_ght=False,
    )
    run_experiment(PatchTSTForForecasting, baseline_cfg, dataset_name, input_len, epochs)

    print(f"\n===== {dataset_name} L={input_len}: GHT Rule ratio=0.75 k=8 ({epochs} epochs) =====")
    ght_cfg = PatchTSTConfig(
        input_len=input_len,
        output_len=24,
        num_features=7,
        intermediate_size=128,
        use_ght=True,
        ght_ratio=0.75,
        ght_k=8,
        ght_assign_hidden=128,
        ght_use_gnn=True,
        ght_local_window=None,
        ght_temperature=1.0,
        ght_graph_mode="rule",
        ght_dual_stream=False,
    )
    run_experiment(PatchTSTForForecastingGHT, ght_cfg, dataset_name, input_len, epochs)


if __name__ == "__main__":
    main()
