import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# -----------------------------
# Construct the results table
# -----------------------------
data = {
    ("QASPER", "F1"):      [55.44, 44.47, 36.15],
    ("QASPER", "ROUGE-L"): [52.93, 45.24, 39.54],
    ("NarrativeQA", "F1"):      [58.03, 54.05, 56.79],
    ("NarrativeQA", "ROUGE-L"): [56.39, 53.98, 55.76],
    ("TriviaQA", "F1"):      [84.13, 85.41, 83.58],
    ("TriviaQA", "ROUGE-L"): [83.61, 84.58, 83.65],
}
index = ["SFR", "BGE", "BM25"]
df = pd.DataFrame(data, index=index)
df.columns = [f"{ds}_{metric}" for ds, metric in df.columns]

# -----------------------------
# 1. Heatmap
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 3))
im = ax.imshow(df.values)  # default colormap is used

# axis ticks and labels
ax.set_xticks(np.arange(df.shape[1]))
ax.set_xticklabels(df.columns, rotation=45, ha="right")
ax.set_yticks(np.arange(df.shape[0]))
ax.set_yticklabels(df.index)

# annotate cells with values
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        ax.text(j, i, f"{df.values[i, j]:.1f}", ha="center", va="center", fontsize=8)

ax.set_title("Performance Heatmap (Higher is better)")
fig.tight_layout()
plt.show()

# -----------------------------
# 2. Radar Chart
# -----------------------------
def make_radar_chart(df, title):
    categories = list(df.columns)
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
    for retriever in df.index:
        values = df.loc[retriever].tolist()
        values += values[:1]  # complete the loop
        ax.plot(angles, values, linewidth=1)  # default colors used
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, y=1.1)
    fig.tight_layout()
    plt.show()

make_radar_chart(df, "Retriever Comparison (Radar Plot)")

# -----------------------------
# 3. Parallel Coordinates Plot
# -----------------------------
# Prepare data for parallel coordinates: add 'Retriever' column
df_pc = df.reset_index().rename(columns={'index': 'Retriever'})
fig, ax = plt.subplots(figsize=(10, 4))
parallel_coordinates(df_pc, class_column='Retriever', ax=ax)
ax.set_title("Parallel Coordinates Plot of Retriever Performance")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
plt.show()


print("Done")