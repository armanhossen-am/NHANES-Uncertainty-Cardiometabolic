from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

PROCESSED = PROJECT / "data" / "processed"
MODELING = PROJECT / "data" / "modeling"
MODELING.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED / "nhanes_cardiometabolic_outcomes_2011_2014.csv"

OUTPUT_FILE = MODELING / "model_ready_2011_2014.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Adult cohort
df = df[df["age"] >= 18].copy()

outcomes = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

predictors = [
    "age",
    "sex",
    "race_ethnicity",
    "education",
    "income_poverty_ratio",
    "bmi",
    "waist_circumference",
    "systolic_bp",
    "diastolic_bp",
    "hba1c",
    "fasting_glucose",
    "total_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "serum_creatinine",
    "albumin_creatinine_ratio",
]

keep_cols = ["seqn", "cycle", "mec_weight", "psu", "strata"] + predictors + outcomes
keep_cols = [c for c in keep_cols if c in df.columns]

df = df[keep_cols].copy()

# Remove rows with missing outcomes
df = df.dropna(subset=outcomes)

# Simple missing handling for first modeling version
for col in predictors:
    if col not in df.columns:
        continue

    if df[col].dtype == "object":
        df[col] = df[col].fillna("Missing")
    else:
        df[col] = df[col].fillna(df[col].median())

# Save model-ready file
df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Shape:", df.shape)
print("\nOutcome prevalence:")
print(df[outcomes].mean())