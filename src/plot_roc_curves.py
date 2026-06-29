from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

PRED_DIR = PROJECT / "results" / "predictions"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_LABELS = {
    "undiagnosed_diabetes": "Undiagnosed diabetes",
    "undiagnosed_hypertension": "Undiagnosed hypertension",
    "undiagnosed_dyslipidemia": "Undiagnosed dyslipidemia",
    "possible_ckd_risk": "Possible CKD risk",
    "any_latent_cardiometabolic_disease": "Any latent disease",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
}

SCENARIO = "scenario_3"  # main figure: routine clinical assessment

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

fig, axes = plt.subplots(3, 2, figsize=(11, 12))
axes = axes.flatten()

for ax, (outcome, label) in zip(axes, OUTCOME_LABELS.items()):
    for model, model_label in MODEL_LABELS.items():
        file = PRED_DIR / f"{model}_{SCENARIO}_{outcome}.csv"

        if not file.exists():
            print("Missing:", file)
            continue

        df = pd.read_csv(file)

        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = roc_auc_score(df["y_true"], df["y_prob"])

        ax.plot(fpr, tpr, linewidth=2, label=f"{model_label} (AUROC={auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_title(label, fontweight="bold", loc="left")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

# Remove empty sixth panel
axes[-1].axis("off")

fig.suptitle(
    "Figure 5. Receiver operating characteristic curves under routine clinical assessment",
    fontsize=15,
    fontweight="bold",
    y=0.995
)

fig.text(
    0.01,
    0.01,
    "ROC curves compare logistic regression and XGBoost using Scenario 3 predictors. "
    "AUROC values are shown in the legend for each outcome.",
    fontsize=9,
    color="#4B5563"
)

plt.tight_layout(rect=[0, 0.035, 1, 0.96])

plt.savefig(FIG_DIR / "figure_5_roc_curves_scenario_3.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_5_roc_curves_scenario_3.pdf", bbox_inches="tight")
plt.close()

print("Saved Figure 5 to:", FIG_DIR)