from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from evaluation import evaluate_model


PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

SCENARIO_DIR = PROJECT / "data" / "modeling" / "scenarios"
RESULTS = PROJECT / "results"
PRED_DIR = RESULTS / "predictions"

RESULTS.mkdir(exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

OUTCOMES = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

BASE_COLS = ["seqn", "cycle", "mec_weight", "psu", "strata"]


def build_preprocessor(X):
    categorical = [c for c in ["sex", "race_ethnicity", "education"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main():
    all_results = []

    for scenario_file in sorted(SCENARIO_DIR.glob("*_model_ready.csv")):
        scenario_name = scenario_file.stem.replace("_model_ready", "")
        df = pd.read_csv(scenario_file, low_memory=False)

        drop_cols = BASE_COLS + OUTCOMES
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y_all = df[OUTCOMES]

        for outcome in OUTCOMES:
            y = y_all[outcome]

            if y.nunique() < 2:
                print(f"Skipping {scenario_name} | {outcome}: only one class")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

            pos = y_train.sum()
            neg = len(y_train) - pos
            scale_pos_weight = neg / pos if pos > 0 else 1

            model = Pipeline(
                [
                    ("preprocess", build_preprocessor(X)),
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=3,
                            learning_rate=0.03,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            scale_pos_weight=scale_pos_weight,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

            print(f"Training XGBoost | {scenario_name} | {outcome}")
            model.fit(X_train, y_train)

            prob = model.predict_proba(X_test)[:, 1]

            pred_df = pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "outcome": outcome,
                    "model": "xgboost",
                    "y_true": y_test.values,
                    "y_prob": prob,
                }
            )

            pred_file = PRED_DIR / f"xgboost_{scenario_name}_{outcome}.csv"
            pred_df.to_csv(pred_file, index=False)

            row = evaluate_model(
                y_true=y_test,
                y_prob=prob,
                model_name="xgboost",
                scenario=scenario_name,
                outcome=outcome,
                n_bootstrap=500,
            )

            row["n_train"] = len(y_train)
            row["n_predictors"] = X.shape[1]
            row["scale_pos_weight"] = scale_pos_weight

            all_results.append(row)

    results_df = pd.DataFrame(all_results)
    output_file = RESULTS / "xgboost_by_scenario.csv"
    results_df.to_csv(output_file, index=False)

    print(results_df)
    print("\nSaved:", output_file)
    print("Predictions saved to:", PRED_DIR)


if __name__ == "__main__":
    main()