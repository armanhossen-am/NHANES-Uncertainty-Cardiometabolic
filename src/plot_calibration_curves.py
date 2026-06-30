from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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

MODELS = {
    "logistic_regression":"Logistic regression",
    "xgboost":"XGBoost",
}

SCENARIO="scenario_3"

plt.rcParams.update({
    "font.family":"Arial",
    "font.size":11,
    "axes.titlesize":13,
    "axes.labelsize":12,
    "savefig.dpi":300,
})

fig,axes=plt.subplots(3,2,figsize=(11.5,12.5))
axes=axes.flatten()

for i,outcome in enumerate(OUTCOMES):

    ax=axes[i]

    for model in MODELS:

        file=PRED_DIR/f"{model}_{SCENARIO}_{outcome}.csv"

        df=pd.read_csv(file)

        frac_pos, mean_pred = calibration_curve(
            df.y_true,
            df.y_prob,
            n_bins=10,
            strategy="quantile"
        )

        brier=brier_score_loss(df.y_true,df.y_prob)

        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2.5,
            color=COLORS[model],
            label=f"{MODELS[model]}\nBrier={brier:.3f}"
        )

    ax.plot(
        [0,1],
        [0,1],
        "--",
        color="gray",
        linewidth=1.2
    )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.set_title(
        OUTCOME_LABELS[outcome],
        loc="left",
        fontweight="bold"
    )

    ax.grid(alpha=.25)

    if i in [0,2,4]:
        ax.set_ylabel("Observed probability")

    if i in [4]:
        ax.set_xlabel("Predicted probability")

    ax.legend(frameon=False,fontsize=8)

axes[-1].axis("off")

fig.suptitle(
    "Figure 7. Calibration curves under routine clinical assessment",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.01,
    0.01,
    "Calibration curves compare predicted and observed event probabilities. "
    "The dashed diagonal represents perfect calibration.",
    fontsize=9
)

plt.tight_layout(rect=[0,0.04,1,.96])

plt.savefig(
    FIG_DIR/"figure_7_calibration.png",
    bbox_inches="tight"
)

plt.savefig(
    FIG_DIR/"figure_7_calibration.pdf",
    bbox_inches="tight"
)

plt.savefig(
    FIG_DIR/"figure_7_calibration.svg",
    bbox_inches="tight"
)

print("Calibration figure saved.")