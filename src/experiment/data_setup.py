import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dataset import DatasetCausalInference
from experiment.paths import ResolvedPaths
from raw_embedding import RawEmbedding


def prepare_causal_inference_dataset(
    cfg: DictConfig,
    paths: ResolvedPaths,
    device: torch.device,
) -> DatasetCausalInference:
    exp = cfg.experiment
    data_root = str(paths.data_root)

    tfm = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
    test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
    train_dataset_no_transform = datasets.FashionMNIST(data_root, train=True, download=True)
    test_dataset_no_transform = datasets.FashionMNIST(data_root, train=False, download=True)

    train_dataset = Subset(train_dataset, range(int(exp.n_train_fMNIST)))
    test_dataset = Subset(test_dataset, range(int(exp.n_test_fMNIST)))

    train_loader = DataLoader(
        train_dataset, batch_size=int(exp.batch_size_autoencoder), shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(exp.batch_size_autoencoder), shuffle=False
    )

    ra = OmegaConf.to_container(exp.raw_autoencoder, resolve=True)
    assert isinstance(ra, dict)
    raw_embedding = RawEmbedding(
        hidden_dim=int(exp.dim_covariate_image),
        train_loader=train_loader,
        test_loader=test_loader,
        device=str(device),
        epochs=int(ra["epochs"]),
        lr=float(ra["lr"]),
        weight_decay=float(ra["weight_decay"]),
    )
    train_embeddings, test_embeddings = raw_embedding.obtain_embeddings()

    torch.save((train_embeddings, test_embeddings), paths.embedding_file)
    train_embeddings, test_embeddings = torch.load(paths.embedding_file)

    return DatasetCausalInference(
        int(exp.dim_covariate),
        int(exp.dim_covariate_image),
        int(exp.dim_post_treatment),
        train_embeddings,
        test_embeddings,
        train_dataset_no_transform,
        test_dataset_no_transform,
    )
