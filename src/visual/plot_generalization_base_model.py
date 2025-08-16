import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

# ---------------- 1. Data Prep ----------------
metrics = ["QASPER F1", "QASPER Rouge-L", "NarrativeQA F1", "NarrativeQA Rouge-L", "TriviaQA F1", "TriviaQA Rouge-L"]
df = pd.DataFrame({
    "Metric": metrics,
    "Mistral7B+RAG": [22.73, 16.71, 40.23, 40.16, 58.43, 49.07],
    "Mistral7B+SARA": [38.83, 41.52, 69.46, 68.02, 85.08, 83.85],
    "MistralSmall+RAG": [30.01, 26.13, 32.89, 27.46, 69.64, 60.10],
    "MistralSmall+SARA": [45.12, 37.10, 58.06, 57.58, 85.35, 78.23],
    "MistralNemo+RAG": [29.93, 20.26, 32.89, 27.46, 66.89, 60.16],
    "MistralNemo+SARA": [46.45, 44.13, 55.69, 53.97, 76.34, 66.55],
    "Llama3.1-8B+RAG": [27.38, 19.93, 43.66, 41.58, 49.81, 38.98],
    "Llama3.1-8B+SARA": [52.56, 49.37, 59.27, 57.20, 85.08, 83.85],
    "Gemma3+RAG": [18.04, 17.11, 8.99, 11.43, 44.3, 35.56],
    "Gemma3+SARA": [22.28, 21.03, 13.01, 12.20, 52.06, 38.27],
})

df[['Dataset', 'Measure']] = df['Metric'].str.split(' ', expand=True)

base_models = ["Mistral7B", "MistralSmall", "MistralNemo", "Llama3.1-8B", "Gemma3"]
model_colors = {
    "Mistral7B": "#3C5488", 
    "MistralSmall": "#DC0000", 
    "MistralNemo": "#00A087", 
    "Llama3.1-8B": "#E64B35", 
    "Gemma3": "#7E6148"
}

# ---------------- Global Style ----------------
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "axes.labelsize": 12, "axes.titlesize": 14, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2,
    "figure.dpi": 300, "text.color": "#222222", "axes.labelcolor": "#222222"
})



# ==============================================================================
# Plot 2: Dumbbell Plot (Cleveland Dot Plot) for F1 across specific datasets
# Highly effective for showing the exact "Delta"
# ==============================================================================
datasets = ["QASPER", "NarrativeQA", "TriviaQA"]
fig2, axes = plt.subplots(1, 3, figsize=(8, 3.5), sharey=True)

# Prepare data for F1 specifically
f1_df = df[df['Measure'] == 'F1'].set_index('Dataset')

for i, dataset in enumerate(datasets):
    ax = axes[i]
    y_positions = np.arange(len(base_models))
    
    for j, model in enumerate(base_models):
        val_rag = f1_df.loc[dataset, f'{model}+RAG']
        val_sara = f1_df.loc[dataset, f'{model}+SARA']
        color = model_colors[model]
        
        # Draw connecting line (the "barbell")
        ax.plot([val_rag, val_sara], [j, j], color='gray', lw=2, zorder=1, alpha=0.5)
        
        # Plot RAG point (empty circle)
        ax.scatter(val_rag, j, color='white', edgecolor=color, s=80, zorder=2, lw=2)
        # Plot SARA point (filled circle)
        ax.scatter(val_sara, j, color=color, s=80, zorder=2, edgecolor='black', lw=0.5)
        
        # Add a text label showing the +Delta
        delta = val_sara - val_rag
        if delta > 0:
            ax.text(val_sara + 2, j, f"+{delta:.1f}", va='center', ha='left', fontsize=9, color=color, fontweight='bold')
            
    ax.set_yticks(y_positions)
    if i == 0:
        ax.set_yticklabels(base_models, fontweight='bold')
    
    ax.set_title(dataset, fontweight='bold')
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='-', alpha=0.2)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

# Add custom legend to the middle plot
empty_dot = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='gray', markeredgewidth=2, linestyle='None', markersize=8, label='RAG')
filled_dot = mlines.Line2D([], [], color='gray', marker='o', markeredgecolor='black', linestyle='None', markersize=8, label='SARA')
fig2.legend(handles=[empty_dot, filled_dot], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

fig2.suptitle("SARA vs. RAG across Models (F1-Scores)", y=1.1, fontsize=12 fontweight='bold')
plt.tight_layout()
fig2.savefig('plot_dumbbell.png', bbox_inches='tight')
plt.close(fig2)

print("Plots successfully generated.")