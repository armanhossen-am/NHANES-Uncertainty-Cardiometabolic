from pathlib import Path
import numpy as np
import pandas as pd


PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")
INTERIM = PROJECT / "data" / "interim"
PROCESSED = PROJECT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INTERIM / "nhanes_master_2011_2014.csv"
OUTPUT_FILE = PROCESSED / "nhanes_cardiometabolic_outcomes_2011_2014.csv"


def mean_existing(df, cols):
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return np.nan
    return df[existing].mean(axis=1, skipna=True)


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    keep = {
        # ID and cycle
        "SEQN": "seqn",
        "cycle": "cycle",

        # demographics
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",
        "RIDRETH1": "race_ethnicity",
        "DMDEDUC2": "education",
        "INDFMPIR": "income_poverty_ratio",

        # survey design
        "WTMEC2YR": "mec_weight",
        "SDMVPSU": "psu",
        "SDMVSTRA": "strata",

        # body and BP
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_circumference",
        "BPXSY1": "sbp1",
        "BPXSY2": "sbp2",
        "BPXSY3": "sbp3",
        "BPXDI1": "dbp1",
        "BPXDI2": "dbp2",
        "BPXDI3": "dbp3",

        # labs
        "LBXGH": "hba1c",
        "LBXGLU": "fasting_glucose",
        "LBXTC": "total_cholesterol",
        "LBDHDD": "hdl_cholesterol",
        "LBXTR": "triglycerides",
        "LBXSCR": "serum_creatinine",
        "URXUMA": "urine_albumin",
        "URXUCR": "urine_creatinine",

        # questionnaires
        "DIQ010": "doctor_told_diabetes",
        "BPQ020": "doctor_told_hypertension",
        "BPQ080": "doctor_told_high_cholesterol",
    }

    available = {k: v for k, v in keep.items() if k in df.columns}
    missing = [k for k in keep if k not in df.columns]

    print("Available:", len(available))
    print("Missing:", missing)

    data = df[list(available.keys())].rename(columns=available)

    # Blood pressure average
    data["systolic_bp"] = mean_existing(data, ["sbp1", "sbp2", "sbp3"])
    data["diastolic_bp"] = mean_existing(data, ["dbp1", "dbp2", "dbp3"])

    # Known disease from questionnaire
    data["known_diabetes"] = np.where(data.get("doctor_told_diabetes") == 1, 1, 0)
    data["known_hypertension"] = np.where(data.get("doctor_told_hypertension") == 1, 1, 0)
    data["known_high_cholesterol"] = np.where(data.get("doctor_told_high_cholesterol") == 1, 1, 0)

    # 1. Diabetes / dysglycemia
    data["biomarker_diabetes"] = np.where(
        (data.get("hba1c") >= 6.5) | (data.get("fasting_glucose") >= 126),
        1,
        0,
    )

    data["undiagnosed_diabetes"] = np.where(
        (data["known_diabetes"] == 0) & (data["biomarker_diabetes"] == 1),
        1,
        0,
    )

    # 2. Hypertension
    data["measured_hypertension"] = np.where(
        (data["systolic_bp"] >= 130) | (data["diastolic_bp"] >= 80),
        1,
        0,
    )

    data["undiagnosed_hypertension"] = np.where(
        (data["known_hypertension"] == 0) & (data["measured_hypertension"] == 1),
        1,
        0,
    )

    # 3. Dyslipidemia
    data["measured_dyslipidemia"] = np.where(
        (data.get("total_cholesterol") >= 240)
        | (data.get("hdl_cholesterol") < 40)
        | (data.get("triglycerides") >= 200),
        1,
        0,
    )

    data["undiagnosed_dyslipidemia"] = np.where(
        (data["known_high_cholesterol"] == 0) & (data["measured_dyslipidemia"] == 1),
        1,
        0,
    )

    # 4. Kidney risk / CKD proxy
    data["albumin_creatinine_ratio"] = np.where(
        data.get("urine_creatinine") > 0,
        data.get("urine_albumin") / data.get("urine_creatinine"),
        np.nan,
    )

    data["possible_ckd_risk"] = np.where(
        (data["albumin_creatinine_ratio"] >= 30),
        1,
        0,
    )

    # 5. Combined latent cardiometabolic burden
    outcome_cols = [
        "undiagnosed_diabetes",
        "undiagnosed_hypertension",
        "undiagnosed_dyslipidemia",
        "possible_ckd_risk",
    ]

    data["latent_cardiometabolic_count"] = data[outcome_cols].sum(axis=1)
    data["any_latent_cardiometabolic_disease"] = np.where(
        data["latent_cardiometabolic_count"] >= 1,
        1,
        0,
    )

    data.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("Shape:", data.shape)
    print("\nOutcome prevalence:")
    print(data[outcome_cols + ["any_latent_cardiometabolic_disease"]].mean())


if __name__ == "__main__":
    main()
