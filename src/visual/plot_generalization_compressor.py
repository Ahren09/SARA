import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FONT_SIZE = 12
plt.rcParams['font.family'] = 'Times New Roman'


# Prepare the data in long format
data = [
    ("QASPER",     "SFR",    "F1",       40.55),
    ("QASPER",     "SFR",    "ROUGE-L",  41.71),
    ("QASPER",     "Stella", "F1",       50.74),
    ("QASPER",     "Stella", "ROUGE-L",  49.33),
    ("QASPER",     "gte",    "F1",       50.16),
    ("QASPER",     "gte",    "ROUGE-L",  48.71),
    ("QASPER",     "Linq",   "F1",       34.87),
    ("QASPER",     "Linq",   "ROUGE-L",  34.01),
    ("QASPER",     "RAG",    "F1",       22.73),
    ("QASPER",     "RAG",    "ROUGE-L",  16.71),

    ("NarrativeQA","SFR",    "F1",       69.15),
    ("NarrativeQA","SFR",    "ROUGE-L",  66.55),
    ("NarrativeQA","Stella", "F1",       59.01),
    ("NarrativeQA","Stella", "ROUGE-L",  57.41),
    ("NarrativeQA","gte",    "F1",       57.76),
    ("NarrativeQA","gte",    "ROUGE-L",  57.25),
    ("NarrativeQA","Linq",   "F1",       40.73),
    ("NarrativeQA","Linq",   "ROUGE-L",  39.59),
    ("NarrativeQA","RAG",    "F1",       40.23),
    ("NarrativeQA","RAG",    "ROUGE-L",  40.16),

    ("TriviaQA",   "SFR",    "F1",       84.74),
    ("TriviaQA",   "SFR",    "ROUGE-L",  84.17),
    ("TriviaQA",   "Stella", "F1",       83.09),
    ("TriviaQA",   "Stella", "ROUGE-L",  79.81),
    ("TriviaQA",   "gte",    "F1",       83.15),
    ("TriviaQA",   "gte",    "ROUGE-L",  79.76),
    ("TriviaQA",   "Linq",   "F1",       56.68),
    ("TriviaQA",   "Linq",   "ROUGE-L",  49.12),
    ("TriviaQA",   "RAG",    "F1",       58.43),
    ("TriviaQA",   "RAG",    "ROUGE-L",  49.07),
]

df = pd.DataFrame(data, columns=["Dataset", "Compressor", "Metric", "Score"])

# Set Seaborn style
# sns.set(style="whitegrid", context="talk")

# Create the figure with 3 subplots (one per dataset)
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

for ax, dataset in zip(axes, df["Dataset"].unique()):
    subset = df[df["Dataset"] == dataset]
    sns.barplot(
        data=subset,
        x="Metric",
        y="Score",
        hue="Compressor",
        ax=ax,
        # edgecolor="black"
        palette=["#ffb5a7", "#fcd5ce", "#f8edeb", "#f9dcc4", "#fec89a"]
    )
    ax.set_title(dataset, fontsize=FONT_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")  # show y‑label only once to save space
    # Move legend to the first subplot only
    if dataset != "QASPER":
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig("outputs/visual/generalization_compressor.pdf", bbox_inches='tight')
plt.show()
