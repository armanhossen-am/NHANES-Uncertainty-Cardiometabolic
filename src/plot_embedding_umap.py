from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import umap

PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

EMBED_FILE = (
    PROJECT
    / "results"
    / "embeddings"
    / "multitask_residual_mlp_embeddings_scenario_3.csv"
)

FIG_DIR = PROJECT / "paper" / "Paper_1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(EMBED_FILE)

embedding_cols = [c for c in df.columns if c.startswith("embedding_")]

X = df[embedding_cols]

reducer = umap.UMAP(
    n_neighbors=25,
    min_dist=0.20,
    metric="euclidean",
    random_state=42,
)

umap_embedding = reducer.fit_transform(X)

df["UMAP1"] = umap_embedding[:, 0]
df["UMAP2"] = umap_embedding[:, 1]

disease = (
    df["undiagnosed_diabetes"]
    + df["undiagnosed_hypertension"]
    + df["undiagnosed_dyslipidemia"]
    + df["possible_ckd_risk"]
)

plt.figure(figsize=(8, 7))

scatter = plt.scatter(
    df["UMAP1"],
    df["UMAP2"],
    c=disease,
    cmap="viridis",
    s=14,
    alpha=0.75,
)

plt.colorbar(scatter, label="Number of latent diseases")

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.title(
    "Figure 9. UMAP visualization of learned patient embeddings",
    fontsize=15,
    weight="bold",
)

plt.tight_layout()

plt.savefig(FIG_DIR / "figure_9_umap_embeddings.png", dpi=300)
plt.savefig(FIG_DIR / "figure_9_umap_embeddings.pdf")
plt.savefig(FIG_DIR / "figure_9_umap_embeddings.svg")

print("Saved Figure 9.")