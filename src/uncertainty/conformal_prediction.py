from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

PRED_DIR = PROJECT / "results" / "predictions"
OUT_DIR = PROJECT / "results" / "conformal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "multitask_mlp",
    "multitask_residual_mlp",
]

SCENARIO = "scenario_3"

OUTCOMES = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level, method="higher")


def prediction_set_binary(p, qhat):
    include_0 = p <= qhat
    include_1 = (1 - p) <= qhat

    if include_0 and include_1:
        return "{0,1}", 2
    if include_1:
        return "{1}", 1
    if include_0:
        return "{0}", 1
    return "{}", 0


def main():
    summaries = []
    all_predictions = []

    for model in MODELS:
        for outcome in OUTCOMES:
            pred_file = PRED_DIR / f"{model}_{SCENARIO}_{outcome}.csv"

            if not pred_file.exists():
                print("Missing:", pred_file.name)
                continue

            df = pd.read_csv(pred_file)

            calib_df, eval_df = train_test_split(
                df,
                test_size=0.50,
                random_state=42,
                stratify=df["y_true"],
            )

            scores = np.where(
                calib_df["y_true"] == 1,
                1 - calib_df["y_prob"],
                calib_df["y_prob"],
            )

            qhat = conformal_quantile(scores, ALPHA)

            sets = []
            sizes = []
            covered = []

            for _, row in eval_df.iterrows():
                pred_set, size = prediction_set_binary(row["y_prob"], qhat)

                is_covered = (
                    ("1" in pred_set) if row["y_true"] == 1 else ("0" in pred_set)
                )

                sets.append(pred_set)
                sizes.append(size)
                covered.append(is_covered)

            eval_df = eval_df.copy()
            eval_df["conformal_set"] = sets
            eval_df["set_size"] = sizes
            eval_df["covered"] = covered
            eval_df["qhat"] = qhat
            eval_df["alpha"] = ALPHA
            eval_df["model"] = model
            eval_df["scenario"] = SCENARIO
            eval_df["outcome"] = outcome

            summaries.append({
                "model": model,
                "scenario": SCENARIO,
                "outcome": outcome,
                "alpha": ALPHA,
                "target_coverage": 1 - ALPHA,
                "empirical_coverage": np.mean(covered),
                "average_set_size": np.mean(sizes),
                "uncertainty_rate": np.mean(np.array(sizes) == 2),
                "singleton_rate": np.mean(np.array(sizes) == 1),
                "empty_rate": np.mean(np.array(sizes) == 0),
                "qhat": qhat,
                "n_calibration": len(calib_df),
                "n_evaluation": len(eval_df),
            })

            all_predictions.append(eval_df)

    summary_df = pd.DataFrame(summaries)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    summary_df.to_csv(OUT_DIR / "conformal_summary_all_models_scenario_3.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "conformal_predictions_all_models_scenario_3.csv", index=False)

    print(summary_df)
    print("\nSaved conformal summary and predictions.")


if __name__ == "__main__":
    main()