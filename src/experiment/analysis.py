from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")


def summarize_and_print(result_pickle: Path) -> pd.DataFrame:
    df_result = pd.read_pickle(result_pickle)

    df_summary = (
        df_result.groupby(["method", "estimator"])
        .agg(
            train_err_mean=("train_err", "mean"),
            train_err_sd=("train_err", np.std),
            test_err_mean=("test_err", "mean"),
            test_err_sd=("test_err", np.std),
        )
        .reset_index()
    )

    method_order = ["biased", "naive", "debiased"]
    estimator_order = ["regression", "ipw", "dr"]

    df_summary["method"] = pd.Categorical(
        df_summary["method"], categories=method_order, ordered=True
    )
    df_summary["estimator"] = pd.Categorical(
        df_summary["estimator"], categories=estimator_order, ordered=True
    )

    df_summary = df_summary.sort_values(["method", "estimator"]).reset_index(drop=True)

    df_summary_dr = (
        df_summary[df_summary["estimator"] == "dr"]
        .drop(columns=["estimator"])
        .reset_index(drop=True)
    )

    print(df_summary_dr)
    df_summary_dr_latex = df_summary_dr.to_latex(index=False)
    print(df_summary_dr_latex)
    return cast(pd.DataFrame, df_summary_dr)
