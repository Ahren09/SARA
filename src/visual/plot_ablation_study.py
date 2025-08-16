import matplotlib.pyplot as plt
import seaborn as sns
import math

results_f1 = {
    "QASPER": [40.55, 39.85, 38.75, 37.39, 32.95, 34.18],
    "NarrativeQA": [69.15, 68.87, 68.93, 66.67, 62.26, 61.75],
    "TriviaQA": [84.74, 84.55, 83.37, 79.12, 76.18, 77.87]
}
methods = ["SARA", "-L", "-R", "-C", "-I", "-P"]

colors = ["#E64B35"] + ["#4DBBD5"] * 5

fig, axes = plt.subplots(1, 3, figsize=(6, 3.5))
plt.subplots_adjust(wspace=0.25)

for idx, (dataset, scores) in enumerate(results_f1.items()):
    ax = axes[idx]
    
    sns.barplot(
        x=methods, 
        y=scores, 
        ax=ax, 
        palette=colors, 
        edgecolor='black', 
        linewidth=0.9
    )
    
    ax.set_title(dataset, fontsize=14, fontweight='bold', pad=10)
    
    ymin = math.floor(min(scores)) - 2
    ymax = math.ceil(max(scores)) + 1
    ax.set_ylim(ymin, ymax)
    
    if idx == 0:
        ax.set_ylabel("F1-Score", fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel("")
        
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10, labelrotation=30)
    
    # --- Trimming the Spines ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set spine width
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 1. Get the actual y-ticks that are visible
    yticks = [t for t in ax.get_yticks() if t >= ymin and t <= ymax]
    
    # 2. Bound the left spine to exactly the min and max tick
    if yticks:
        ax.spines['left'].set_bounds(min(yticks), max(yticks))
        
    # 3. Bound the bottom spine to the first and last bar
    ax.spines['bottom'].set_bounds(0, len(methods) - 1)
    
    # Optional: Push the spines outward slightly (Tufte style)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('outputs/visual/ablation.pdf', dpi=300, bbox_inches='tight')
