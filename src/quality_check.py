from pathlib import Path
import pandas as pd

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED / "nhanes_cardiometabolic_outcomes_2011_2014.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)

outcome_cols = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

print("\nOutcome prevalence:")
print(df[outcome_cols].mean().sort_values(ascending=False))

missing = df.isna().mean().sort_values(ascending=False)
missing.to_csv(RESULTS / "missingness_summary.csv")

outcome_summary = df[outcome_cols].mean().reset_index()
outcome_summary.columns = ["outcome", "prevalence"]
outcome_summary.to_csv(RESULTS / "outcome_prevalence.csv", index=False)

print("\nTop missing variables:")
print(missing.head(20))

print("\nSaved:")
print(RESULTS / "missingness_summary.csv")
print(RESULTS / "outcome_prevalence.csv")