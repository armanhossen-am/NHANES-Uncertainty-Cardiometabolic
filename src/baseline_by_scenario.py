from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

SCENARIO_DIR = PROJECT / "data" / "modeling" / "scenarios"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

outcomes = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

base_cols = ["seqn", "cycle", "mec_weight", "psu", "strata"]

all_results = []

for scenario_file in sorted(SCENARIO_DIR.glob("*_model_ready.csv")):
    scenario_name = scenario_file.stem.replace("_model_ready", "")
    df = pd.read_csv(scenario_file, low_memory=False)

    drop_cols = base_cols + outcomes
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y_all = df[outcomes]

    categorical = [c for c in ["sex", "race_ethnicity", "education"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ]
    )

    for outcome in outcomes:
        y = y_all[outcome]

        if y.nunique() < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        model = Pipeline([
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        all_results.append({
            "scenario": scenario_name,
            "outcome": outcome,
            "model": "logistic_regression",
            "auroc": roc_auc_score(y_test, prob),
            "auprc": average_precision_score(y_test, prob),
            "f1": f1_score(y_test, pred),
            "prevalence_test": y_test.mean(),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_predictors": X.shape[1],
        })

results_df = pd.DataFrame(all_results)
output_file = RESULTS / "baseline_logistic_by_scenario.csv"
results_df.to_csv(output_file, index=False)

print(results_df)
print("\nSaved:", output_file)