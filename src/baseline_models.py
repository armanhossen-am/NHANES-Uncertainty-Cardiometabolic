from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

DATA = PROJECT / "data" / "modeling" / "model_ready_2011_2014.csv"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

outcomes = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

drop_cols = ["seqn", "cycle", "mec_weight", "psu", "strata"] + outcomes

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y_all = df[outcomes]

categorical = ["sex", "race_ethnicity", "education"]
categorical = [c for c in categorical if c in X.columns]

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

results = []

for outcome in outcomes:
    y = y_all[outcome]

    if y.nunique() < 2:
        print(f"Skipping {outcome}: only one class")
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

    results.append({
        "outcome": outcome,
        "model": "logistic_regression",
        "auroc": roc_auc_score(y_test, prob),
        "auprc": average_precision_score(y_test, prob),
        "f1": f1_score(y_test, pred),
        "prevalence_test": y_test.mean()
    })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS / "baseline_logistic_results.csv", index=False)

print(results_df)
print("\nSaved:", RESULTS / "baseline_logistic_results.csv")