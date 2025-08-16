import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text
# Create output directory
os.makedirs("outputs/visual", exist_ok=True)

# 1. Data Setup
data = {
    "Base LLM": [
        "Mistral-7B", "Mistral-7B",
        "Mistral-Nemo", "Mistral-Nemo",
        "Mistral-Small", "Mistral-Small",
        "Llama-3.1", "Llama-3.1",
        "Gemma-3", "Gemma-3"
    ],
    "Method": ["RAG", "SARA"] * 5,
    "Relevance": [65.58, 83.29, 65.17, 84.82, 62.01, 85.02, 56.32, 85.79, 32.37, 39.20],
    "Correctness": [26.90, 40.62, 34.34, 41.66, 43.59, 43.76, 37.05, 44.16, 16.93, 22.75],
    "Semantic Similarity": [62.28, 77.81, 56.94, 77.38, 65.54, 79.92, 68.13, 79.23, 43.62, 47.92],
    "Faithfulness": [60.61, 63.54, 55.05, 59.51, 63.51, 65.19, 63.52, 63.49, 49.87, 54.29],
}
df = pd.DataFrame(data)

# Aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.0

COLOR_RAG = '#8F9CA3'   # Gray
COLOR_SARA = '#D73027'  # Red
MODELS = ["Gemma-3", "Llama-3.1", "Mistral-Small", "Mistral-Nemo", "Mistral-7B"]
METRICS = ["Relevance", "Correctness", "Semantic Similarity", "Faithfulness"]

fig, axes = plt.subplots(1, 4, figsize=(9, 2.5), sharey=True)
fig.subplots_adjust(wspace=0.15)

texts = []

FONTSIZE = 10
for i, metric in enumerate(METRICS):
    ax = axes[i]
    
    # Uniform horizontal grid lines for readability
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#CCCCCC')
    
    for y_idx, model in enumerate(MODELS):
        rag_val = df[(df["Base LLM"] == model) & (df["Method"] == "RAG")][metric].values[0]
        sara_val = df[(df["Base LLM"] == model) & (df["Method"] == "SARA")][metric].values[0]
        
        # Connecting line
        ax.plot([rag_val, sara_val], [y_idx, y_idx], color='#444444', alpha=0.4, lw=2, zorder=1)
        
        # Dots
        ax.scatter(rag_val, y_idx, color=COLOR_RAG, s=80, zorder=2, edgecolors='white', linewidth=0.8)
        ax.scatter(sara_val, y_idx, color=COLOR_SARA, s=80, zorder=3, edgecolors='white', linewidth=0.8)
        
        # Add legend labels once
        if i == 0 and y_idx == len(MODELS)-1:
            texts.append(ax.text(rag_val, y_idx + 0.4, "RAG", ha='center', va='center', fontsize=FONTSIZE, color=COLOR_RAG, fontweight='bold'))
            texts.append(ax.text(sara_val, y_idx + 0.4, "SARA", ha='center', va='center', fontsize=FONTSIZE, color=COLOR_SARA, fontweight='bold'))

    # UNIFORM SCALE: Applying 0-100 or 10-95 to all subplots
    # ax.set_xlim(0, 100)
    # ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(10, 90)
    ax.set_xticks([10, 30, 50, 70, 90])
    
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=FONTSIZE)
    ax.set_title(metric, fontsize=FONTSIZE + 1, fontweight='bold', pad=10)
    
    # Clean up axis borders
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    
    
adjust_text(texts, ax=ax, #expand_text=(1.05, 1.2)
            arrowprops=None)
plt.tight_layout()
fig.savefig("outputs/visual/generalization_base_model_llm_metrics.pdf", dpi=300, bbox_inches='tight')
plt.show()