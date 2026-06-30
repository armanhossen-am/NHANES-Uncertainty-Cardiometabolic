from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

CONFORMAL_DIR = PROJECT / "results" / "conformal"
TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FILE = CONFORMAL_DIR / "conformal_summary_random_forest_scenario_3.csv"

df = pd.read_csv(FILE)

df["outcome"] = df["outcome"].replace({
    "undiagnosed_diabetes": "Undiagnosed diabetes",
    "undiagnosed_hypertension": "Undiagnosed hypertension",
    "undiagnosed_dyslipidemia": "Undiagnosed dyslipidemia",
    "possible_ckd_risk": "Possible CKD risk",
    "any_latent_cardiometabolic_disease": "Any latent cardiometabolic disease",
})

table = df[
    [
        "outcome",
        "target_coverage",
        "empirical_coverage",
        "average_set_size",
        "uncertainty_rate",
        "singleton_rate",
        "empty_rate",
        "n_calibration",
        "n_evaluation",
    ]
].copy()

table.columns = [
    "Outcome",
    "Target coverage",
    "Empirical coverage",
    "Average set size",
    "Uncertainty rate",
    "Singleton rate",
    "Empty rate",
    "Calibration N",
    "Evaluation N",
]

for col in [
    "Target coverage",
    "Empirical coverage",
    "Average set size",
    "Uncertainty rate",
    "Singleton rate",
    "Empty rate",
]:
    table[col] = table[col].round(3)

table.to_csv(TABLE_DIR / "table_3_conformal_prediction.csv", index=False)

fig_height = max(4, len(table) * 0.55)
fig, ax = plt.subplots(figsize=(13, fig_height))
ax.axis("off")

tbl = ax.table(
    cellText=table.values,
    colLabels=table.columns,
    loc="center",
    cellLoc="left",
    colLoc="left",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.5)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    cell.set_linewidth(0.4)

    if row == 0:
        cell.set_facecolor("#E8F1FA")
        cell.set_text_props(weight="bold")
    else:
        cell.set_facecolor("#FFFFFF" if row % 2 else "#F9FAFB")

ax.set_title(
    "Table 3. Conformal prediction performance under routine clinical assessment",
    fontsize=14,
    fontweight="bold",
    pad=18,
)

ax.text(
    0,
    -0.08,
    "Prediction sets were generated using split conformal prediction with target coverage of 90%. "
    "Uncertainty rate represents the proportion of non-singleton prediction sets.",
    transform=ax.transAxes,
    fontsize=9,
    color="#4B5563",
)

plt.tight_layout()

plt.savefig(TABLE_DIR / "table_3_conformal_prediction.png", dpi=300, bbox_inches="tight")
plt.savefig(TABLE_DIR / "table_3_conformal_prediction.pdf", bbox_inches="tight")
plt.close()

print("Saved Table 3 to:", TABLE_DIR)