from pathlib import Path
import pandas as pd

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

DATA = PROJECT / "data" / "modeling" / "model_ready_2011_2014.csv"
META = PROJECT / "docs" / "metadata" / "VARIABLE_DICTIONARY.csv"
OUT = PROJECT / "data" / "modeling" / "scenarios"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)
meta = pd.read_csv(META)

outcomes = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

base_cols = ["seqn", "cycle", "mec_weight", "psu", "strata"]

# Map original NHANES variables to cleaned names
name_map = {
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH1": "race_ethnicity",
    "DMDEDUC2": "education",
    "INDFMPIR": "income_poverty_ratio",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_circumference",
    "BPXSY1": "systolic_bp",
    "BPXDI1": "diastolic_bp",
    "LBXTC": "total_cholesterol",
    "LBDHDD": "hdl_cholesterol",
    "LBXTR": "triglycerides",
    "LBXSCR": "serum_creatinine",
    "LBXGH": "hba1c",
    "LBXGLU": "fasting_glucose",
    "URXUMA": "urine_albumin",
    "URXUCR": "urine_creatinine",
    "DIQ010": "doctor_told_diabetes",
    "BPQ020": "doctor_told_hypertension",
    "BPQ080": "doctor_told_high_cholesterol",
}

for scenario in ["Scenario_1", "Scenario_2", "Scenario_3"]:
    selected = meta.loc[meta[scenario] == "Yes", "Variable"].tolist()
    selected_clean = [name_map[v] for v in selected if v in name_map]

    cols = base_cols + selected_clean + outcomes
    cols = [c for c in cols if c in df.columns]

    scenario_df = df[cols].copy()

    output_file = OUT / f"{scenario.lower()}_model_ready.csv"
    scenario_df.to_csv(output_file, index=False)

    print(f"{scenario}: saved {output_file}")
    print("Shape:", scenario_df.shape)
    print("Predictors:", selected_clean)
    print()