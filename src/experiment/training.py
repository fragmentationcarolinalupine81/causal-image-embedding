from collections.abc import Iterable, Sized
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from causal_embedding import DebiasedEmbeddingNet
from naive_embedding import NaiveEmbeddingNet


def train_naive_embedding_net(
    model: NaiveEmbeddingNet,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_n: int,
    print_loss: bool,
    desc: str | None,
) -> None:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    epoch_iter: Iterable[int] = range(epochs)
    if desc is not None:
        epoch_iter = tqdm(range(epochs), desc=desc)
    for epoch in epoch_iter:
        model.train()
        loss_each_epoch = 0.0
        for batch in train_loader:
            x, d, v, y = batch
            x, d, v, y = x.to(device), d.to(device), v.to(device), y.to(device)
            _x_v, hat_v = model(x, d, v, y)
            optimizer.zero_grad()
            loss = mse(hat_v, v)
            loss.backward()
            optimizer.step()
            loss_each_epoch += loss.item() * x.size(0)
        if print_loss:
            print(f"Epoch {epoch + 1}/{epochs} Loss: {loss_each_epoch / train_n:.4f}")


def train_debiased_embedding_net(
    model: DebiasedEmbeddingNet,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_n: int,
    print_loss: bool,
) -> None:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_each_epoch = 0.0
        for batch in train_loader:
            x, d, v, y = batch
            x, d, v, y = x.to(device), d.to(device), v.to(device), y.to(device)
            _x_v, p_v, hat_p_v, hat_d, hat_y, hat_v = model(x, d, v, y)
            optimizer.zero_grad()
            loss = mse(hat_v, v) + bce(hat_d, d) + mse(hat_p_v, p_v) + mse(hat_y, y)
            loss.backward()
            optimizer.step()
            loss_each_epoch += loss.item() * x.size(0)
        if print_loss:
            print(f"Epoch {epoch + 1}/{epochs} Loss: {loss_each_epoch / train_n:.4f}")


def dataloader_dataset_len(loader: DataLoader) -> int:
    return len(cast(Sized, loader.dataset))
