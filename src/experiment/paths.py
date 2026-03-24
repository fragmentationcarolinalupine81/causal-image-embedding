from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ResolvedPaths:
    data_root: Path
    embedding_file: Path
    result_pickle: Path


def resolve_paths(cfg: DictConfig, base_dir: Path | None = None) -> ResolvedPaths:
    root = base_dir if base_dir is not None else Path.cwd()
    p = OmegaConf.to_container(cfg.paths, resolve=True)
    assert isinstance(p, dict)
    return ResolvedPaths(
        data_root=(root / str(p["data_root"])).resolve(),
        embedding_file=(root / str(p["embedding_file"])).resolve(),
        result_pickle=(root / str(p["result_pickle"])).resolve(),
    )
