"""
QASPER sensitivity plot:
- One PDF only
- Left: F1
- Right: ROUGE-L
- Overlays:
    1. Mistral7B_QA_Compress_Stage0
    2. Mistral7B
"""

import os
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
# ===== [Configuration] =====
DPI = 600
FILETYPE = "pdf"

COLOR_STAGE0 = "#E64B35"
COLOR_BASE = "#3C5488"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

MODEL_STYLES = {
    "Mistral7B_QA_Compress_Stage0": {
        "color": COLOR_STAGE0,
        "marker": "o",
        "label": "Mistral7B QA Compress Stage0",
    },
    "Mistral7B": {
        "color": COLOR_BASE,
        "marker": "s",
        "label": "Mistral7B",
    },
}

model_name_map = {
    "Mistral7B_QA_Compress_Stage0": "w/ Compression",
    "Mistral7B": "no Compression",
}

# ===== [Data Loading] =====
def load_data(dataset_name, model_name, n, plot_type="mixed"):
    excel_path = "outputs/qa/performance_summary.xlsx"
    if not os.path.exists(excel_path):
        print(f"Missing file: {excel_path}")
        return None

    try:
        df = pd.read_excel(excel_path, sheet_name=dataset_name)
        df = df[df["model_name"] == model_name].copy()

        if plot_type == "mixed":
            df = df[df["n"] == n]
        else:
            df = df[df["k"] == df["n"]]

        if df.empty:
            print(f"No data for {dataset_name} / {model_name} / {plot_type}")
            return None

        return df.sort_values(by="k").reset_index(drop=True)
    except Exception as e:
        print(f"Error loading {dataset_name} ({model_name}): {e}")
        return None

def _style_ax(ax):
    ax.grid(True, axis="y", ls="-", color="#E4E4E4", lw=0.5, zorder=0)

# ===== [Plotting] =====
def plot_qasper_two_panel(n, out_dir):
    dataset_name = "qasper"
    model_configs = [
        {"model": "Mistral7B_QA_Compress_Stage0", "type": "mixed"},
        {"model": "Mistral7B", "type": "natural_only"},
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)

    plotted_any = False
    all_k = set()

    for cfg in model_configs:
        df = load_data(dataset_name, cfg["model"], n, cfg["type"])
        if df is None or df.empty:
            continue
        
        df["model_name"] = df["model_name"].map(model_name_map)

        style = MODEL_STYLES[cfg["model"]]
        k = df["k"].tolist()
        all_k.update(k)

        ax1.plot(
            k, df["f1"],
            marker=style["marker"], ms=4.5,
            color=style["color"], mfc="white",
            lw=1.2, label=style["label"]
        )

        ax2.plot(
            k, df["rougel"],
            marker=style["marker"], ms=4.5,
            color=style["color"], mfc="white",
            lw=1.2, label=style["label"]
        )

        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("No valid QASPER data found.")
        return

    xticks = sorted(all_k)
    
    
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    # ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax1.set_ylim(28, 55)

    ax1.set_title("QASPER: F1 Score")
    ax1.set_xlabel("Natural-language evidence, k")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(xticks)
    _style_ax(ax1)

    ax2.set_title("QASPER: ROUGE-L")
    ax2.set_xlabel("Natural-language evidence, k")
    ax2.set_xticks(xticks)
    _style_ax(ax2)

    # one shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    save_path = Path(out_dir) / f"qasper_compare_n{n}.{FILETYPE}"
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved -> {save_path}")

# ===== [Main] =====
def main():
    n = 10
    out_dir = f"outputs/visual/qasper_compare_n{n}"
    os.makedirs(out_dir, exist_ok=True)
    plot_qasper_two_panel(n, out_dir)

if __name__ == "__main__":
    main()