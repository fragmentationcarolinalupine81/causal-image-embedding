import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from causal_embedding import DebiasedEmbeddingNet
from causal_inference import ATE, compute_ATE
from dataset import ObservedDataset
from experiment.data_setup import prepare_causal_inference_dataset
from experiment.embedding_utils import compute_covariate_image_embeddings
from experiment.paths import resolve_paths
from experiment.results import build_result_rows
from experiment.seeding import set_all_seeds
from experiment.training import (
    dataloader_dataset_len,
    train_debiased_embedding_net,
    train_naive_embedding_net,
)
from naive_embedding import NaiveEmbeddingNet
from visualize import visualize_dataset


def _compute_ground_truth_ate_and_estimators(
    dataset: dict,
    naive_image_embeddings: torch.Tensor,
    debiased_image_embeddings: torch.Tensor,
) -> ATE:
    true_ates = compute_ATE(dataset, ate_type="true")
    true_ate = float(true_ates.dr)
    biased_ates = compute_ATE(dataset, ate_type="biased")
    naive_ates = compute_ATE(
        dataset,
        ate_type="learned_covariate_image",
        covariate_image=naive_image_embeddings,
    )
    debiased_ates = compute_ATE(
        dataset,
        ate_type="learned_covariate_image",
        covariate_image=debiased_image_embeddings,
    )
    return ATE(true_ate, biased_ates, naive_ates, debiased_ates)


def run_experiment(cfg: DictConfig) -> None:
    set_all_seeds(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = resolve_paths(cfg)
    exp = cfg.experiment

    dim_naive_embed = int(exp.dim_covariate_image_embed) + int(exp.dim_post_treatment_embed)

    dataset_ci = prepare_causal_inference_dataset(cfg, paths, device)

    df_result = pd.DataFrame(columns=["id", "estimator", "method", "train_err", "test_err"])
    num_seeds = int(exp.num_seeds)

    for seed in tqdm(range(num_seeds)):
        training_dataset_ci = dataset_ci.generate_dataset(int(exp.training_sample_size), train=True)
        test_dataset_ci = dataset_ci.generate_dataset(int(exp.test_sample_size), train=False)

        if exp.display_image:
            visualize_dataset(training_dataset_ci, max_size=3)
            visualize_dataset(test_dataset_ci, max_size=3)

        observed_train = ObservedDataset(
            training_dataset_ci["covariate"],
            training_dataset_ci["treatment"],
            training_dataset_ci["post_treatment_image_dataset"],
            training_dataset_ci["outcome"],
        )
        observed_test = ObservedDataset(
            test_dataset_ci["covariate"],
            test_dataset_ci["treatment"],
            test_dataset_ci["post_treatment_image_dataset"],
            test_dataset_ci["outcome"],
        )

        train_loader_ci = DataLoader(
            observed_train,
            batch_size=int(exp.batch_size_causal_embedding),
            shuffle=True,
        )
        test_loader_ci = DataLoader(
            observed_test,
            batch_size=int(exp.batch_size_causal_embedding),
            shuffle=False,
        )
        train_n_ci = dataloader_dataset_len(train_loader_ci)

        naive_net = NaiveEmbeddingNet(
            int(exp.dim_covariate),
            dim_naive_embed,
            int(exp.dim_post_treatment_embed),
        )
        train_naive_embedding_net(
            naive_net,
            train_loader_ci,
            device=device,
            epochs=int(exp.epochs_embed),
            lr=float(exp.lr_embed),
            weight_decay=float(exp.weight_decay_embed),
            train_n=train_n_ci,
            print_loss=bool(exp.print_loss),
            desc=f"Seed {seed} / {num_seeds} naive",
        )

        debiased_net = DebiasedEmbeddingNet(
            int(exp.dim_covariate),
            int(exp.dim_covariate_image_embed),
            int(exp.dim_post_treatment_embed),
        )
        train_debiased_embedding_net(
            debiased_net,
            train_loader_ci,
            device=device,
            epochs=int(exp.epochs_embed),
            lr=float(exp.lr_embed),
            weight_decay=float(exp.weight_decay_embed),
            train_n=train_n_ci,
            print_loss=bool(exp.print_loss),
        )

        naive_train_emb = compute_covariate_image_embeddings(
            train_loader_ci, naive_net, dim_naive_embed, device
        )
        naive_test_emb = compute_covariate_image_embeddings(
            test_loader_ci, naive_net, dim_naive_embed, device
        )
        deb_train_emb = compute_covariate_image_embeddings(
            train_loader_ci,
            debiased_net,
            int(exp.dim_covariate_image_embed),
            device,
        )
        deb_test_emb = compute_covariate_image_embeddings(
            test_loader_ci,
            debiased_net,
            int(exp.dim_covariate_image_embed),
            device,
        )

        train_ates = _compute_ground_truth_ate_and_estimators(
            training_dataset_ci,
            naive_train_emb,
            deb_train_emb,
        )
        test_ates = _compute_ground_truth_ate_and_estimators(
            test_dataset_ci,
            naive_test_emb,
            deb_test_emb,
        )

        new_rows = build_result_rows(seed, train_ates, test_ates)
        df_new = pd.DataFrame(new_rows)
        if exp.print_result_per_seed:
            print(df_new)
        df_result = pd.concat([df_result, df_new], ignore_index=True)

    df_result.to_pickle(paths.result_pickle)
