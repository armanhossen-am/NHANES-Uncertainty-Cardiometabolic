from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

PRED_DIR = PROJECT / "results" / "predictions"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def net_benefit(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pred_positive = y_prob >= threshold

    tp = np.sum((pred_positive == 1) & (y_true == 1))
    fp = np.sum((pred_positive == 1) & (y_true == 0))
    n = len(y_true)

    if threshold >= 1:
        return np.nan

    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def treat_all_net_benefit(y_true, threshold):
    prevalence = np.mean(y_true)
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))


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

thresholds = np.linspace(0.01, 0.50, 100)

fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5))
axes = axes.flatten()

for idx, outcome in enumerate(OUTCOMES):
    ax = axes[idx]
    y_ref = None

    for model in MODELS:
        file = PRED_DIR / f"{model}_{SCENARIO}_{outcome}.csv"

        if not file.exists():
            print("Missing:", file)
            continue

        df = pd.read_csv(file)
        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        y_ref = y_true

        nb = [net_benefit(y_true, y_prob, t) for t in thresholds]

        ax.plot(
            thresholds,
            nb,
            linewidth=2.5,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )

    if y_ref is not None:
        treat_all = [treat_all_net_benefit(y_ref, t) for t in thresholds]
        treat_none = np.zeros_like(thresholds)

        ax.plot(
            thresholds,
            treat_all,
            linestyle="--",
            linewidth=1.3,
            color="#7A7A7A",
            label="Treat all",
        )

        ax.plot(
            thresholds,
            treat_none,
            linestyle=":",
            linewidth=1.3,
            color="#444444",
            label="Treat none",
        )

    ax.set_title(OUTCOME_LABELS[outcome], fontweight="bold", loc="left")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

axes[-1].axis("off")

fig.suptitle(
    "Figure 8. Decision curve analysis under routine clinical assessment",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.01,
    0.012,
    "Decision curves compare the clinical net benefit of logistic regression and XGBoost across threshold probabilities. "
    "Treat-all and treat-none strategies are shown as reference approaches.",
    fontsize=9.5,
    color="#4B5563",
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

plt.savefig(FIG_DIR / "figure_8_decision_curve.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_8_decision_curve.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_8_decision_curve.svg", bbox_inches="tight")
plt.close()

print("Saved Figure 8 to:", FIG_DIR)