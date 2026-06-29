from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"
FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(TABLE_DIR / "table_2_model_performance.csv")

# Clean labels
data["scenario_label"] = data["scenario"].replace({
    "scenario_1": "Community\nscreening",
    "scenario_2": "Primary care\nscreening",
    "scenario_3": "Routine clinical\nassessment",
})

data["model_label"] = data["model"].replace({
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
})

data["outcome_label"] = data["outcome"].replace({
    "undiagnosed_diabetes": "Undiagnosed diabetes",
    "undiagnosed_hypertension": "Undiagnosed hypertension",
    "undiagnosed_dyslipidemia": "Undiagnosed dyslipidemia",
    "possible_ckd_risk": "Possible CKD risk",
    "any_latent_cardiometabolic_disease": "Any latent disease",
})

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

outcomes = data["outcome_label"].unique()

fig, axes = plt.subplots(
    nrows=len(outcomes),
    ncols=1,
    figsize=(9, 13),
    sharex=True
)

if len(outcomes) == 1:
    axes = [axes]

for ax, outcome in zip(axes, outcomes):
    sub = data[data["outcome_label"] == outcome]

    pivot = sub.pivot_table(
        index="scenario_label",
        columns="model_label",
        values="auroc"
    )

    pivot = pivot.reindex([
        "Community\nscreening",
        "Primary care\nscreening",
        "Routine clinical\nassessment"
    ])

    x = np.arange(len(pivot.index))
    width = 0.36

    ax.bar(
        x - width/2,
        pivot["Logistic regression"],
        width,
        label="Logistic regression",
        edgecolor="black",
        linewidth=0.5
    )

    ax.bar(
        x + width/2,
        pivot["XGBoost"],
        width,
        label="XGBoost",
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_title(outcome, fontweight="bold", loc="left")
    ax.set_ylim(0.45, 1.00)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for i, scenario in enumerate(pivot.index):
        for j, model in enumerate(["Logistic regression", "XGBoost"]):
            value = pivot.loc[scenario, model]
            xpos = x[i] + (-width/2 if model == "Logistic regression" else width/2)
            ax.text(
                xpos,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)

axes[-1].set_xlabel("Deployment scenario")
for ax in axes:
    ax.set_ylabel("AUROC")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

fig.suptitle(
    "Figure 4. Model discrimination across deployment scenarios",
    fontsize=15,
    fontweight="bold",
    y=0.995
)

fig.text(
    0.01,
    0.01,
    "AUROC = area under the receiver operating characteristic curve. "
    "Scenario 1 uses survey-only variables; Scenario 2 adds physical examination variables; "
    "Scenario 3 adds routine laboratory variables.",
    fontsize=9,
    color="#4B5563"
)

plt.tight_layout(rect=[0, 0.035, 1, 0.965])

plt.savefig(FIG_DIR / "figure_4_auroc_by_scenario_model.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "figure_4_auroc_by_scenario_model.pdf", bbox_inches="tight")
plt.close()

print("Saved Figure 4 to:", FIG_DIR)