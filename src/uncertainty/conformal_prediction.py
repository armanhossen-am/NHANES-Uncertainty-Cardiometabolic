from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")
sys.path.append(str(PROJECT / "src"))

PRED_DIR = PROJECT / "results" / "predictions"
RESULTS = PROJECT / "results"
OUT_DIR = RESULTS / "conformal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "random_forest"
SCENARIO = "scenario_3"
ALPHA = 0.10  # 90% target coverage

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


def make_prediction_set(p, qhat):
    include_0 = p <= qhat
    include_1 = (1 - p) <= qhat

    if include_0 and include_1:
        return "{0,1}", 2
    elif include_1:
        return "{1}", 1
    elif include_0:
        return "{0}", 1
    else:
        return "{}", 0


def main():
    all_summary = []
    all_predictions = []

    for outcome in OUTCOMES:
        file = PRED_DIR / f"{MODEL}_{SCENARIO}_{outcome}.csv"

        if not file.exists():
            print("Missing:", file)
            continue

        df = pd.read_csv(file)

        calib_df, eval_df = train_test_split(
            df,
            test_size=0.50,
            random_state=42,
            stratify=df["y_true"],
        )

        # nonconformity score = 1 - probability assigned to true class
        calib_scores = np.where(
            calib_df["y_true"] == 1,
            1 - calib_df["y_prob"],
            calib_df["y_prob"],
        )

        qhat = conformal_quantile(calib_scores, ALPHA)

        eval_sets = []
        set_sizes = []
        covered = []

        for _, row in eval_df.iterrows():
            pred_set, size = make_prediction_set(row["y_prob"], qhat)

            if row["y_true"] == 1:
                is_covered = "1" in pred_set
            else:
                is_covered = "0" in pred_set

            eval_sets.append(pred_set)
            set_sizes.append(size)
            covered.append(is_covered)

        eval_df = eval_df.copy()
        eval_df["conformal_set"] = eval_sets
        eval_df["set_size"] = set_sizes
        eval_df["covered"] = covered
        eval_df["qhat"] = qhat
        eval_df["alpha"] = ALPHA

        coverage = np.mean(covered)
        avg_set_size = np.mean(set_sizes)
        uncertainty_rate = np.mean(np.array(set_sizes) == 2)
        singleton_rate = np.mean(np.array(set_sizes) == 1)
        empty_rate = np.mean(np.array(set_sizes) == 0)

        all_summary.append({
            "model": MODEL,
            "scenario": SCENARIO,
            "outcome": outcome,
            "alpha": ALPHA,
            "target_coverage": 1 - ALPHA,
            "empirical_coverage": coverage,
            "average_set_size": avg_set_size,
            "uncertainty_rate": uncertainty_rate,
            "singleton_rate": singleton_rate,
            "empty_rate": empty_rate,
            "qhat": qhat,
            "n_calibration": len(calib_df),
            "n_evaluation": len(eval_df),
        })

        eval_df["outcome"] = outcome
        all_predictions.append(eval_df)

    summary_df = pd.DataFrame(all_summary)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    summary_file = OUT_DIR / f"conformal_summary_{MODEL}_{SCENARIO}.csv"
    pred_file = OUT_DIR / f"conformal_predictions_{MODEL}_{SCENARIO}.csv"

    summary_df.to_csv(summary_file, index=False)
    pred_df.to_csv(pred_file, index=False)

    print(summary_df)
    print("\nSaved:", summary_file)
    print("Saved:", pred_file)


if __name__ == "__main__":
    main()