from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
}

COLORS = {
    "logistic_regression": "#0072B2",
    "xgboost": "#D55E00",
}

SCENARIO = "scenario_3"

perf = pd.read_csv(PERF_FILE)

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
    prevalence_drawn = False

    for model in ["logistic_regression", "xgboost"]:
        file = PRED_DIR / f"{model}_{SCENARIO}_{outcome}.csv"

        if not file.exists():
            print("Missing:", file)
            continue

        df = pd.read_csv(file)

        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        ap = average_precision_score(df["y_true"], df["y_prob"])
        prevalence = df["y_true"].mean()

        sub = perf[
            (perf["scenario_key"] == SCENARIO)
            & (perf["model_key"] == model)
            & (perf["outcome_key"] == outcome)
        ]

        if not sub.empty and "AUPRC (95% CI)" in sub.columns:
            ap_label = sub["AUPRC (95% CI)"].iloc[0]
        else:
            ap_label = f"{ap:.2f}"

        ax.plot(
            recall,
            precision,
            linewidth=2.8,
            color=COLORS[model],
            label=f"{MODEL_LABELS[model]}\nAUPRC {ap_label}",
        )

        if not prevalence_drawn:
            ax.axhline(
                prevalence,
                linestyle="--",
                linewidth=1.2,
                color="#B0B0B0",
                label=f"Prevalence = {prevalence:.2f}",
            )
            prevalence_drawn = True

    ax.set_title(OUTCOME_LABELS[outcome], fontweight="bold", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)

    if idx in [0, 2, 4]:
        ax.set_ylabel("Precision")
    else:
        ax.set_ylabel("")

    if idx in [4]:
        ax.set_xlabel("Recall")
    else:
        ax.set_xlabel("")

    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

axes[-1].axis("off")

fig.suptitle(
    "Figure 6. Precision–recall curves under routine clinical assessment",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.01,
    0.012,
    "Precision–recall curves compare logistic regression and XGBoost using Scenario 3 predictors. "
    "AUPRC values include bootstrap 95% confidence intervals; dashed lines indicate test-set prevalence.",
    fontsize=9.5,
    color="#4B5563",
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

plt.savefig(FIG_DIR / "figure_6_pr_curves_scenario_3.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_6_pr_curves_scenario_3.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_6_pr_curves_scenario_3.svg", bbox_inches="tight")
plt.close()

print("Saved Figure 6 to:", FIG_DIR)