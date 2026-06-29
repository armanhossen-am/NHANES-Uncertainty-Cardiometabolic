from pathlib import Path
import pandas as pd

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

RESULTS = PROJECT / "results"
TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

logistic_file = RESULTS / "baseline_logistic_by_scenario.csv"
xgb_file = RESULTS / "xgboost_by_scenario.csv"

logistic = pd.read_csv(logistic_file)
xgb = pd.read_csv(xgb_file)

combined = pd.concat([logistic, xgb], ignore_index=True)

# Round values for manuscript table
table = combined.copy()
for col in ["auroc", "auprc", "f1", "prevalence_test"]:
    if col in table.columns:
        table[col] = table[col].round(3)

table = table[
    [
        "scenario",
        "outcome",
        "model",
        "auroc",
        "auprc",
        "f1",
        "prevalence_test",
        "n_test",
        "n_predictors",
    ]
]

table = table.sort_values(["scenario", "outcome", "model"])

output_csv = TABLE_DIR / "table_2_model_performance.csv"
table.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print(table)