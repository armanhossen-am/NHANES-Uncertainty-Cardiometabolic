from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

INTERIM = PROJECT / "data" / "interim"
PROCESSED = PROJECT / "data" / "processed"
MODELING = PROJECT / "data" / "modeling"

FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
TABLE_DIR = PROJECT / "paper" / "Paper_1" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

master = pd.read_csv(INTERIM / "nhanes_master_2011_2014.csv", low_memory=False)
outcomes = pd.read_csv(PROCESSED / "nhanes_cardiometabolic_outcomes_2011_2014.csv", low_memory=False)
model_ready = pd.read_csv(MODELING / "model_ready_2011_2014.csv", low_memory=False)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# =====================================================
# Figure 1: Overall Study Workflow
# =====================================================

steps = [
    ("NHANES 2011–2014", "Demographics, questionnaire, examination, and laboratory data"),
    ("Data integration", "Merge survey components by participant identifier"),
    ("Outcome definition", "Define undiagnosed cardiometabolic outcomes"),
    ("Deployment scenarios", "Community, primary care, and routine clinical assessment"),
    ("Model development", "Baseline models and multi-task deep learning"),
    ("Uncertainty quantification", "Conformal prediction and prediction sets"),
    ("Clinical decision layer", "Confident low-risk, confident high-risk, or uncertain/referral"),
]

fig, ax = plt.subplots(figsize=(9, 10))
ax.axis("off")

y = 0.92
colors = ["#E8F1FA", "#EAF7F2", "#FFF4E5", "#F1ECFA", "#EAF4FF", "#F9EAF2", "#EAF7F2"]

for i, (title, subtitle) in enumerate(steps):
    box = FancyBboxPatch(
        (0.10, y - 0.06), 0.80, 0.085,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.2,
        facecolor=colors[i],
        edgecolor="#2F3A4A"
    )
    ax.add_patch(box)

    ax.text(0.14, y - 0.005, f"{i+1}", ha="center", va="center",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="circle", facecolor="white", edgecolor="#2F3A4A"))

    ax.text(0.20, y + 0.008, title, ha="left", va="center",
            fontsize=12, fontweight="bold", color="#1F2933")
    ax.text(0.20, y - 0.028, subtitle, ha="left", va="center",
            fontsize=10, color="#374151")

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(0.50, y - 0.08),
            xytext=(0.50, y - 0.12),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="#2F3A4A")
        )

    y -= 0.13

ax.set_title("Figure 1. Overall study workflow", fontsize=15, fontweight="bold", pad=18)
savefig("figure_1_study_workflow.png")


# =====================================================
# Figure 2: Population Flowchart
# =====================================================

n_raw = len(master)
n_adult = (outcomes["age"] >= 18).sum()
n_model = len(model_ready)

flow_steps = [
    (f"NHANES 2011–2014\nN = {n_raw:,}", ""),
    (f"Adults aged ≥18 years\nN = {n_adult:,}", f"Excluded age <18 years\nN = {n_raw - n_adult:,}"),
    (f"Final analytical cohort\nN = {n_model:,}", f"Excluded during outcome or predictor preprocessing\nN = {n_adult - n_model:,}"),
]

fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.axis("off")

y_positions = [0.78, 0.50, 0.22]

for i, (main_text, side_text) in enumerate(flow_steps):
    main_box = FancyBboxPatch(
        (0.18, y_positions[i]), 0.46, 0.13,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=1.3,
        facecolor="#F8FAFC",
        edgecolor="#1D4E89"
    )
    ax.add_patch(main_box)

    ax.text(0.41, y_positions[i] + 0.065, main_text,
            ha="center", va="center", fontsize=11, fontweight="bold")

    if side_text:
        side_box = FancyBboxPatch(
            (0.72, y_positions[i]), 0.25, 0.13,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            linewidth=1.1,
            facecolor="#FFFFFF",
            edgecolor="#6B7280",
            linestyle="--"
        )
        ax.add_patch(side_box)

        ax.text(0.845, y_positions[i] + 0.065, side_text,
                ha="center", va="center", fontsize=9)

        ax.plot([0.64, 0.72], [y_positions[i] + 0.065, y_positions[i] + 0.065],
                color="#6B7280", linewidth=1)

    if i < len(flow_steps) - 1:
        ax.annotate(
            "",
            xy=(0.41, y_positions[i+1] + 0.15),
            xytext=(0.41, y_positions[i]),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="#1D4E89")
        )

ax.set_title("Figure 2. Study population flowchart", fontsize=15, fontweight="bold", pad=16)
savefig("figure_2_population_flowchart.png")


# =====================================================
# Figure 3: Outcome Prevalence
# =====================================================

outcome_cols = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_dyslipidemia",
    "possible_ckd_risk",
    "any_latent_cardiometabolic_disease",
]

labels = [
    "Undiagnosed\ndiabetes",
    "Undiagnosed\nhypertension",
    "Undiagnosed\ndyslipidemia",
    "Possible\nCKD risk",
    "Any latent\ncardiometabolic\ndisease"
]

prev = model_ready[outcome_cols].mean() * 100

fig, ax = plt.subplots(figsize=(9.5, 5.5))

bars = ax.bar(labels, prev.values, edgecolor="#1F2933", linewidth=0.8)

ax.set_ylabel("Prevalence (%)")
ax.set_title("Figure 3. Prevalence of latent cardiometabolic outcomes", fontsize=15, fontweight="bold", pad=16)
ax.set_ylim(0, max(prev.values) * 1.30)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

for bar, value in zip(bars, prev.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(prev.values) * 0.025,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax.text(
    0, -0.22,
    "Prevalence estimated in the model-ready analytical cohort. Survey-weighted estimates will be added in the final analysis.",
    transform=ax.transAxes,
    fontsize=9,
    color="#4B5563"
)

savefig("figure_3_outcome_prevalence.png")


# =====================================================
# Table 1: Baseline Characteristics
# =====================================================

def mean_sd(series):
    return f"{series.mean():.1f} ({series.std():.1f})"

def median_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{series.median():.2f} ({q1:.2f}–{q3:.2f})"

def n_pct(condition):
    n = condition.sum()
    pct = n / len(condition) * 100
    return f"{n:,} ({pct:.1f})"

table = []

table.append(["Age, years, mean (SD)", mean_sd(model_ready["age"])])
table.append(["Age ≥65 years, n (%)", n_pct(model_ready["age"] >= 65)])
table.append(["Male, n (%)", n_pct(model_ready["sex"] == 1)])
table.append(["Female, n (%)", n_pct(model_ready["sex"] == 2)])

if "bmi" in model_ready.columns:
    table.append(["BMI, kg/m², mean (SD)", mean_sd(model_ready["bmi"])])

if "waist_circumference" in model_ready.columns:
    table.append(["Waist circumference, cm, mean (SD)", mean_sd(model_ready["waist_circumference"])])

if "systolic_bp" in model_ready.columns:
    table.append(["Systolic blood pressure, mmHg, mean (SD)", mean_sd(model_ready["systolic_bp"])])

if "diastolic_bp" in model_ready.columns:
    table.append(["Diastolic blood pressure, mmHg, mean (SD)", mean_sd(model_ready["diastolic_bp"])])

if "income_poverty_ratio" in model_ready.columns:
    table.append(["Poverty-income ratio, median (IQR)", median_iqr(model_ready["income_poverty_ratio"])])

table_df = pd.DataFrame(table, columns=["Characteristic", f"Overall cohort (N = {len(model_ready):,})"])
table_df.to_csv(TABLE_DIR / "table_1_baseline_characteristics.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="left",
    colLoc="left",
    colWidths=[0.62, 0.38]
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.55)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    cell.set_linewidth(0.6)

    if row == 0:
        cell.set_facecolor("#E8F1FA")
        cell.set_text_props(weight="bold", color="#111827")
    else:
        if row % 2 == 0:
            cell.set_facecolor("#F9FAFB")
        else:
            cell.set_facecolor("#FFFFFF")

ax.set_title("Table 1. Baseline characteristics of the analytical cohort",
             fontsize=15, fontweight="bold", pad=18)

ax.text(
    0.0, -0.08,
    "Continuous variables are shown as mean (SD) or median (IQR); categorical variables are shown as n (%).",
    transform=ax.transAxes,
    fontsize=9,
    color="#4B5563"
)

savefig("table_1_baseline_characteristics.png")

print("Publication-style figures saved to:", FIG_DIR)
print("Publication-style table saved to:", TABLE_DIR)