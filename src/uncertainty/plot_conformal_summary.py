from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

CONFORMAL_DIR = PROJECT / "results" / "conformal"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FILE = CONFORMAL_DIR / "conformal_summary_all_models_scenario_3.csv"

df = pd.read_csv(FILE)

outcome_map = {
    "undiagnosed_diabetes": "Diabetes",
    "undiagnosed_hypertension": "Hypertension",
    "undiagnosed_dyslipidemia": "Dyslipidemia",
    "possible_ckd_risk": "CKD risk",
    "any_latent_cardiometabolic_disease": "Any latent disease",
}

model_map = {
    "logistic_regression": "Logistic",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "multitask_mlp": "Multi-task MLP",
    "multitask_residual_mlp": "Residual MTL-MLP",
}

df["Outcome"] = df["outcome"].map(outcome_map)
df["Model"] = df["model"].map(model_map)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

pivot_cov = df.pivot(index="Outcome", columns="Model", values="empirical_coverage")
pivot_unc = df.pivot(index="Outcome", columns="Model", values="uncertainty_rate")

outcome_order = ["Diabetes", "Hypertension", "Dyslipidemia", "CKD risk", "Any latent disease"]
pivot_cov = pivot_cov.reindex(outcome_order)
pivot_unc = pivot_unc.reindex(outcome_order)

pivot_cov.plot(kind="bar", ax=axes[0], width=0.82, edgecolor="black", linewidth=0.3)
axes[0].axhline(0.90, linestyle="--", linewidth=1.2, color="gray", label="Target coverage")
axes[0].set_ylabel("Empirical coverage")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("(A) Empirical coverage", loc="left", fontweight="bold")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, fontsize=8, ncol=3)

pivot_unc.plot(kind="bar", ax=axes[1], width=0.82, edgecolor="black", linewidth=0.3)
axes[1].set_ylabel("Uncertainty rate")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("(B) Non-singleton prediction set rate", loc="left", fontweight="bold")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(frameon=False, fontsize=8, ncol=3)

axes[1].set_xlabel("Outcome")
axes[1].set_xticklabels(outcome_order, rotation=25, ha="right")

fig.suptitle(
    "Figure 10. Conformal prediction performance across models",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.01,
    0.01,
    "Split conformal prediction was applied at 90% target coverage under the routine clinical assessment scenario. "
    "Uncertainty rate represents the proportion of non-singleton prediction sets.",
    fontsize=9,
    color="#4B5563",
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

plt.savefig(FIG_DIR / "figure_10_conformal_summary_all_models.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_10_conformal_summary_all_models.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_10_conformal_summary_all_models.svg", bbox_inches="tight")

print("Saved Figure 10.")