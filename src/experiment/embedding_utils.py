from collections.abc import Sized
from typing import cast

import torch
from torch.utils.data import DataLoader


def compute_covariate_image_embeddings(
    dataloader: DataLoader,
    model: torch.nn.Module,
    dim_covariate_image_embed: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    n = len(cast(Sized, dataloader.dataset))
    out = torch.zeros(n, dim_covariate_image_embed, device=device)
    idx = 0
    for batch in dataloader:
        x, _d, v, _y = batch
        batch_size = x.size(0)
        with torch.no_grad():
            z = model.covariate_image_encoder(v.to(device))  # type: ignore[operator]
        out[idx : idx + batch_size] = z
        idx += batch_size
    return out.cpu()
