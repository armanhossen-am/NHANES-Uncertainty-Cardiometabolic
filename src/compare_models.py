from pathlib import Path
import pandas as pd

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

RESULTS = PROJECT / "results"
TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

files = [
    RESULTS / "baseline_logistic_by_scenario.csv",
    RESULTS / "random_forest_by_scenario.csv",
    RESULTS / "xgboost_by_scenario.csv",
    RESULTS / "lightgbm_by_scenario.csv",
]

dfs = [pd.read_csv(f) for f in files if f.exists()]
combined = pd.concat(dfs, ignore_index=True)

combined["model"] = combined["model"].replace({
    "logistic_regression": "Logistic regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
})

combined["model"] = combined["model"].replace({
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
})

combined["outcome"] = combined["outcome"].replace({
    "undiagnosed_diabetes": "Undiagnosed diabetes",
    "undiagnosed_hypertension": "Undiagnosed hypertension",
    "undiagnosed_dyslipidemia": "Undiagnosed dyslipidemia",
    "possible_ckd_risk": "Possible CKD risk",
    "any_latent_cardiometabolic_disease": "Any latent cardiometabolic disease",
})

combined["AUROC (95% CI)"] = combined.apply(
    lambda r: f"{r['auroc']:.3f} ({r['auroc_ci_low']:.3f}–{r['auroc_ci_high']:.3f})",
    axis=1
)

combined["AUPRC (95% CI)"] = combined.apply(
    lambda r: f"{r['auprc']:.3f} ({r['auprc_ci_low']:.3f}–{r['auprc_ci_high']:.3f})",
    axis=1
)

table = combined[
    [
        "scenario",
        "outcome",
        "model",
        "AUROC (95% CI)",
        "AUPRC (95% CI)",
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "brier_score",
        "prevalence_test",
        "n_test",
        "n_predictors",
    ]
].copy()

for col in [
    "accuracy",
    "recall",
    "specificity",
    "precision",
    "f1",
    "brier_score",
    "prevalence_test",
]:
    table[col] = table[col].round(3)

table = table.sort_values(["scenario", "outcome", "model"])

output_csv = TABLE_DIR / "table_2_model_performance.csv"
table.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print(table)