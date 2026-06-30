from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

CONFORMAL_DIR = PROJECT / "results" / "conformal"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FILE = CONFORMAL_DIR / "conformal_summary_random_forest_scenario_3.csv"

df = pd.read_csv(FILE)

label_map = {
    "undiagnosed_diabetes": "Diabetes",
    "undiagnosed_hypertension": "Hypertension",
    "undiagnosed_dyslipidemia": "Dyslipidemia",
    "possible_ckd_risk": "CKD risk",
    "any_latent_cardiometabolic_disease": "Any latent disease",
}

df["Outcome"] = df["outcome"].map(label_map)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

x = np.arange(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.bar(
    x - width / 2,
    df["empirical_coverage"],
    width,
    label="Empirical coverage",
    edgecolor="black",
    linewidth=0.5,
)

ax.bar(
    x + width / 2,
    df["uncertainty_rate"],
    width,
    label="Uncertainty rate",
    edgecolor="black",
    linewidth=0.5,
)

ax.axhline(
    df["target_coverage"].iloc[0],
    linestyle="--",
    linewidth=1.3,
    color="gray",
    label="Target coverage",
)

ax.set_xticks(x)
ax.set_xticklabels(df["Outcome"], rotation=25, ha="right")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1.05)
ax.set_title(
    "Figure 10. Conformal prediction coverage and uncertainty",
    fontsize=15,
    fontweight="bold",
)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(FIG_DIR / "figure_10_conformal_summary.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_10_conformal_summary.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_10_conformal_summary.svg", bbox_inches="tight")

print("Saved Figure 10 to:", FIG_DIR)