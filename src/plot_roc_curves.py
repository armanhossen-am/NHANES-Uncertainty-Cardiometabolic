from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

PRED_DIR = PROJECT / "results" / "predictions"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)

PERF_FILE = TABLE_DIR / "table_2_model_performance.csv"

OUTCOMES = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

OUTCOME_LABELS = {
    "undiagnosed_diabetes": "(A) Undiagnosed diabetes",
    "undiagnosed_hypertension": "(B) Undiagnosed hypertension",
    "undiagnosed_dyslipidemia": "(C) Undiagnosed dyslipidemia",
    "possible_ckd_risk": "(D) Possible CKD risk",
    "any_latent_cardiometabolic_disease": "(E) Any latent disease",
}

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "multitask_mlp",
    "multitask_residual_mlp",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "multitask_mlp": "MTL-MLP",
    "multitask_residual_mlp": "Residual MTL-MLP",
}

COLORS = {
    "logistic_regression": "#0072B2",
    "random_forest": "#009E73",
    "xgboost": "#D55E00",
    "lightgbm": "#CC79A7",
    "multitask_mlp": "#F0E442",
    "multitask_residual_mlp": "#56B4E9",
}

SCENARIO = "scenario_3"
SCENARIO_LABEL = "Routine clinical assessment"

perf = pd.read_csv(PERF_FILE)

# convert table labels back if needed
perf["scenario_key"] = perf["scenario"].replace({
    "Community screening": "scenario_1",
    "Primary care screening": "scenario_2",
    "Routine clinical assessment": "scenario_3",
})

perf["model_key"] = perf["model"].replace({
    "Logistic regression": "logistic_regression",
    "XGBoost": "xgboost",
})

perf["outcome_key"] = perf["outcome"].replace({
    "Undiagnosed diabetes": "undiagnosed_diabetes",
    "Undiagnosed hypertension": "undiagnosed_hypertension",
    "Undiagnosed dyslipidemia": "undiagnosed_dyslipidemia",
    "Possible CKD risk": "possible_ckd_risk",
    "Any latent cardiometabolic disease": "any_latent_cardiometabolic_disease",
})

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5))
axes = axes.flatten()

for idx, outcome in enumerate(OUTCOMES):
    ax = axes[idx]

    for model in MODELS:
        file = PRED_DIR / f"{model}_{SCENARIO}_{outcome}.csv"

        if not file.exists():
            print("Missing:", file)
            continue

        df = pd.read_csv(file)

        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = roc_auc_score(df["y_true"], df["y_prob"])

        sub = perf[
            (perf["scenario_key"] == SCENARIO)
            & (perf["model_key"] == model)
            & (perf["outcome_key"] == outcome)
        ]

        if not sub.empty and "AUROC (95% CI)" in sub.columns:
            auc_label = sub["AUROC (95% CI)"].iloc[0]
        else:
            auc_label = f"{auc:.2f}"

        ax.plot(
            fpr,
            tpr,
            linewidth=2.8,
            color=COLORS[model],
            label=f"{MODEL_LABELS[model]}\nAUROC {auc_label}",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="#B0B0B0",
    )

    ax.set_title(OUTCOME_LABELS[outcome], fontweight="bold", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)

    if idx in [0, 2, 4]:
        ax.set_ylabel("True positive rate")
    else:
        ax.set_ylabel("")

    if idx in [4]:
        ax.set_xlabel("False positive rate")
    else:
        ax.set_xlabel("")

    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

axes[-1].axis("off")

fig.suptitle(
    "Figure 5. Receiver operating characteristic curves under routine clinical assessment",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.01,
    0.012,
    "ROC curves compare logistic regression and XGBoost using Scenario 3 predictors. "
    "AUROC values include bootstrap 95% confidence intervals.",
    fontsize=9.5,
    color="#4B5563",
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

plt.savefig(FIG_DIR / "figure_5_roc_curves_scenario_3.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_5_roc_curves_scenario_3.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_5_roc_curves_scenario_3.svg", bbox_inches="tight")
plt.close()

print("Saved updated Figure 5 to:", FIG_DIR)