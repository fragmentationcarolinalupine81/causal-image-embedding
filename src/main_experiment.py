from pathlib import Path

import hydra
from omegaconf import DictConfig

from experiment.run import run_experiment


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
