from pathlib import Path

import hydra
from omegaconf import DictConfig

from experiment.analysis import summarize_and_print


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "conf"),
    config_name="analysis",
)
def main(cfg: DictConfig) -> None:
    p = Path(str(cfg.paths.result_pickle))
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    summarize_and_print(p)


if __name__ == "__main__":
    main()
