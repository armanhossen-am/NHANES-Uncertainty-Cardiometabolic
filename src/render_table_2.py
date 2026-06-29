from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"

TABLE_FILE = TABLE_DIR / "table_2_model_performance.csv"

df = pd.read_csv(TABLE_FILE)

# Keep the main manuscript columns only
display_df = df[
    [
        "scenario",
        "outcome",
        "model",
        "AUROC (95% CI)",
        "AUPRC (95% CI)",
        "recall",
        "specificity",
        "f1",
        "brier_score",
    ]
].copy()

display_df.columns = [
    "Scenario",
    "Outcome",
    "Model",
    "AUROC (95% CI)",
    "AUPRC (95% CI)",
    "Sensitivity",
    "Specificity",
    "F1",
    "Brier",
]

for col in ["Sensitivity", "Specificity", "F1", "Brier"]:
    display_df[col] = display_df[col].round(3)

# Save full CSV and manuscript display CSV
display_df.to_csv(TABLE_DIR / "table_2_model_performance_display.csv", index=False)

fig_height = max(8, len(display_df) * 0.32)
fig, ax = plt.subplots(figsize=(16, fig_height))
ax.axis("off")

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="left",
    colLoc="left",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.35)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    cell.set_linewidth(0.4)

    if row == 0:
        cell.set_facecolor("#E8F1FA")
        cell.set_text_props(weight="bold", color="#111827")
    else:
        if row % 2 == 0:
            cell.set_facecolor("#F9FAFB")
        else:
            cell.set_facecolor("#FFFFFF")

ax.set_title(
    "Table 2. Baseline model performance across deployment scenarios",
    fontsize=15,
    fontweight="bold",
    pad=18,
)

ax.text(
    0,
    -0.02,
    "AUROC = area under the receiver operating characteristic curve; "
    "AUPRC = area under the precision-recall curve. "
    "Confidence intervals were estimated using bootstrap resampling.",
    transform=ax.transAxes,
    fontsize=9,
    color="#4B5563",
)

plt.tight_layout()
plt.savefig(TABLE_DIR / "table_2_model_performance.png", dpi=300, bbox_inches="tight")
plt.savefig(TABLE_DIR / "table_2_model_performance.pdf", bbox_inches="tight")
plt.close()

print("Saved Table 2 to:", TABLE_DIR)