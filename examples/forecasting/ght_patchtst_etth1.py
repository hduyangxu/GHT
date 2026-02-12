import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecastingGHT


def main():
    input_len = 96
    output_len = 24

    model_cfg = PatchTSTConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
        intermediate_size=128,
        use_ght=True,
        ght_ratio=0.5,
        ght_k=8,
        ght_assign_hidden=128,
        ght_use_gnn=True,
        ght_local_window=None,
        ght_temperature=1.0,
    )

    cfg = BasicTSForecastingConfig(
        model=PatchTSTForForecastingGHT,
        model_config=model_cfg,
        dataset_name="ETTh1_mini",
        gpus=None,
        num_epochs=1,
        input_len=input_len,
        output_len=output_len,
        lr=0.001,
    )

    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
