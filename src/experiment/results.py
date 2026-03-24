from causal_inference import ATE


def build_result_rows(seed: int, train_ates: ATE, test_ates: ATE) -> list[dict]:
    true_ate = train_ates.true_ATE
    rows: list[dict] = []
    for method, ate_attr in (
        ("biased", "biased_ATE"),
        ("naive", "naive_ATE"),
        ("debiased", "debiased_ATE"),
    ):
        tr = getattr(train_ates, ate_attr)
        te = getattr(test_ates, ate_attr)
        for estimator, err_name in (
            ("regression", "error_reg"),
            ("ipw", "error_ipw"),
            ("dr", "error_dr"),
        ):
            err_tr = getattr(tr, err_name)
            err_te = getattr(te, err_name)
            rows.append(
                {
                    "id": seed,
                    "estimator": estimator,
                    "method": method,
                    "train_err": err_tr(true_ate),
                    "test_err": err_te(true_ate),
                }
            )
    return rows
