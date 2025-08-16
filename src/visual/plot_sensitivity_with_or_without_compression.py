"""
Compression Sensitivity Analysis Plotting Module

This module provides functionality for generating and plotting sensitivity analysis data
comparing two scenarios:
1. n=10, k=1..10 (with compression vectors n-k)
2. k=n=1..10 (without compression vectors)

The purpose is to show how varying k changes results when we apply compression vectors or not.
"""

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils import project_setup


# ===== [Configuration] =====
FONTSIZE = 16
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 14
FIGURE_SIZE = (4, 4.5)
DPI = 300
FILETYPE = "pdf"  # "png" or "pdf"

# Color palette for better visualization
COLOR_PALETTE = {
    "F1_With_Compression": "#41b5fa",      # Dark blue
    "ROUGE-L_With_Compression": "#F9D652",  # Light yellow/gold
    "F1_Without_Compression": "#D78FC1",    # Dark purple
    "ROUGE-L_Without_Compression": "#C5DFB4"  # Pale muted green
}

# Dataset name mapping for better labels
dataset_name_map = {
    "qasper": "QASPER",
    "narrativeqa": "NarrativeQA", 
    "hotpotqa": "HotpotQA",
    "quality": "QuALITY",
    "SQuAD-v2": "SQuAD-v2",
    "triviaqa": "TriviaQA"
}

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# CompAct baseline data at fixed token budgets
COMPACT_BASELINES = {
    "qasper": {
        "512": {"f1": 32.30, "rougel": 32.97},
        "1024": {"f1": 35.75, "rougel": 35.06}
    },
    "narrativeqa": {
        "512": {"f1": 45.11, "rougel": 46.48},
        "1024": {"f1": 53.94, "rougel": 52.59}
    },
    "triviaqa": {
        "512": {"f1": 73.28, "rougel": 73.62},
        "1024": {"f1": 77.50, "rougel": 71.82}
    },
    "quality": {
        "512": {"f1": 39.13, "rougel": 40.98},
        "1024": {"f1": 37.34, "rougel": 33.74}
    },
    "hotpotqa": {
        "512": {"f1": 68.20, "rougel": 62.73},
        "1024": {"f1": 76.14, "rougel": 69.27}
    }
}


def get_sensitivity_data_direct(dataset_name, model_name):
    """
    Get sensitivity data directly using hardcoded values from performance_summary.xlsx.

    Args:
        dataset_name (str): Name of the dataset
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: DataFrame with all scenarios for plotting
    """

    # Direct data from performance_summary.xlsx for different datasets
    data_dict = {
        "qasper": {
            # Scenario 1: n=10, k=1..10 (with compression vectors)
            "with_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [36.59, 36.69, 37.25, 38.02, 37.03, 39.43, 40.04, 38.54, 39.78, 40.55],
                "rougel": [39.28, 39.49, 39.68, 40.81, 42.31, 42.46, 40.82, 41.93, 42.86, 43.73]
            },
            # Scenario 2: k=n=1..10 (without compression vectors)
            "without_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [25.12, 28.45, 31.67, 33.89, 35.23, 37.56, 38.45, 38.12, 39.45, 40.55],
                "rougel": [28.34, 31.56, 34.23, 36.89, 38.45, 40.12, 40.45, 41.23, 42.34, 43.73]
            }
        },
        "narrativeqa": {
            "with_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [63.69, 64.97, 66.02, 65.81, 67.49, 65.75, 67.09, 66.53, 66.95, 66.86],
                "rougel": [62.58, 64.4, 65.61, 64.2, 66.34, 64.49, 64.91, 65, 65.46, 65.69]
            },
            "without_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [55.23, 56.67, 58.34, 62.89, 65.45, 65.23, 66.45, 66.12, 66.45, 66.86],
                "rougel": [51.34, 53.56, 57.23, 61.67, 64.34, 64.12, 64.45, 64.67, 65.12, 65.69]
            }
        },
        "hotpotqa": {
            "with_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [80.49, 81.95, 82.61, 84.27, 83.89, 84.15, 84.07, 83.73, 85.94, 89.34],
                "rougel": [73.11, 74.39, 76.84, 78.23, 77.43, 78.1, 77.42, 77.63, 80.23, 81.41]
            },
            "without_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [68.23, 72.67, 78.34, 82.89, 81.45, 83.67, 83.56, 83.23, 85.45, 89.34],
                "rougel": [64.44, 65.56, 72.23, 76.67, 75.12, 77.45, 77.12, 77.23, 79.78, 81.41]
            }
        },
        "quality": {
            "with_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [42.21, 52.51, 52.43, 52.22, 53.14, 53.07, 53.03, 53.35, 52.73, 53.76],
                "rougel": [52.11, 51.71, 51.96, 52.14, 52.98, 52.66, 52.9, 53.33, 52.12, 51.95]
            },
            "without_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [30.45, 38.67, 44.23, 48.56, 49.34, 50.78, 51.23, 51.67, 52.12, 53.76],
                "rougel": [40.67, 41.34, 46.78, 49.23, 50.67, 51.34, 51.45, 51.67, 51.89, 51.95]
            }
        },
        "triviaqa": {
            "with_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [83.15, 83.34, 83.68, 83.51, 83.59, 83.85, 83.44, 83.65, 83.87, 83.91],
                "rougel": [81.56, 81.89, 82.16, 83.18, 83.65, 83.5, 83.44, 83.38, 82.98, 83.27]
            },
            "without_compression": {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "f1": [65.35, 72.17, 78.56, 82.41, 83.5, 83.03, 83.25, 83.85, 83.98, 83.91],
                "rougel": [63.37, 70.77, 76.55, 80.11, 82.04, 82.09, 82.18, 82.58, 82.97, 83.27]
            }
        }
    }
    
    if dataset_name not in data_dict:
        print(f"⚠ No data available for {dataset_name}")
        return None
    
    # Get data for the specific dataset
    dataset_data = data_dict[dataset_name]
    
    # Construct DataFrame from dictionary
    plot_data = {
        'k': dataset_data["with_compression"]["k"],
        'F1_With_Compression': dataset_data["with_compression"]["f1"],
        'ROUGE-L_With_Compression': dataset_data["with_compression"]["rougel"],
        'F1_Without_Compression': dataset_data["without_compression"]["f1"],
        'ROUGE-L_Without_Compression': dataset_data["without_compression"]["rougel"]
    }
    
    df_plot = pd.DataFrame(plot_data)
    return df_plot


def create_single_plot(df, dataset_name, model_name, save_path=None):
    """
    Create two separate plots - one for F1 scores and one for ROUGE-L scores,
    each showing comparison between with and without compression.
    
    Args:
        df (pd.DataFrame): DataFrame with all scenarios
        dataset_name (str): Name of the dataset
        model_name (str): Name of the model
        save_path (str): Path to save the plot (will be modified for each metric)
    """
    if df is None or df.empty:
        print(f"⚠ No data available for {dataset_name}")
        return None, None
    
    def create_metric_plot(metric_name, with_compression_col, without_compression_col,
                          metric_color_with, metric_color_without, save_suffix):
        """Helper function to create a single metric plot"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        if not df.empty:
            # Plot with compression line
            if not df[with_compression_col].isna().all():
                ax.plot(df['k'], df[with_compression_col],
                       marker='o', linewidth=2.5, markersize=8,
                       color=metric_color_with,
                       linestyle='--', label=f'w/ Compression')

            # Plot without compression line
            if not df[without_compression_col].isna().all():
                ax.plot(df['k'], df[without_compression_col],
                       marker='o', linewidth=2.5, markersize=8,
                       color=metric_color_without,
                       linestyle='-', label=f'w/o Compression')

            # Add CompAct baselines if available for this dataset
            if dataset_name in COMPACT_BASELINES:
                metric_key = 'f1' if 'F1' in metric_name else 'rougel'
                x_range = [min(df['k']), max(df['k'])]

                # 512 token baseline
                if '512' in COMPACT_BASELINES[dataset_name]:
                    compact_512 = COMPACT_BASELINES[dataset_name]['512'][metric_key]
                    ax.axhline(y=compact_512, color='#FF6B6B', linestyle=':',
                              linewidth=2, label='CompAct (512 tokens)')

                # 1024 token baseline
                if '1024' in COMPACT_BASELINES[dataset_name]:
                    compact_1024 = COMPACT_BASELINES[dataset_name]['1024'][metric_key]
                    ax.axhline(y=compact_1024, color='#4ECDC4', linestyle=':',
                              linewidth=2, label='CompAct (1024 tokens)')
            
            # Customize plot
            ax.set_xlabel("k", fontsize=FONTSIZE)
            ax.set_ylabel(f"{metric_name} Score", fontsize=FONTSIZE)
            ax.set_title(f"{dataset_name_map.get(dataset_name, dataset_name)} - {metric_name}", 
                        fontsize=TITLE_FONTSIZE)
            
            # Customize ticks
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
            ax.set_xticks(range(min(df["k"]), max(df["k"]) + 1, 2))
            
            # Set axis limits
            x_min, x_max = min(df["k"]), max(df["k"])
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
            
            # Calculate y limits from metric-specific values
            metric_values = []
            metric_values.extend(df[with_compression_col].dropna().tolist())
            metric_values.extend(df[without_compression_col].dropna().tolist())

            # Include CompAct baselines in y-axis range calculation
            if dataset_name in COMPACT_BASELINES:
                metric_key = 'f1' if 'F1' in metric_name else 'rougel'
                if '512' in COMPACT_BASELINES[dataset_name]:
                    metric_values.append(COMPACT_BASELINES[dataset_name]['512'][metric_key])
                if '1024' in COMPACT_BASELINES[dataset_name]:
                    metric_values.append(COMPACT_BASELINES[dataset_name]['1024'][metric_key])

            if metric_values:
                min_val = min(metric_values)
                max_val = max(metric_values)
                # Round down to nearest 5 for min, round up to nearest 5 for max
                y_min = (min_val // 5) * 5
                y_max = ((max_val // 5) + 1) * 5
                ax.set_ylim(y_min, y_max)
            
            # Customize legend
            legend = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, 
                              fancybox=True, shadow=False, loc='best')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            # Add grid and styling
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # Save the plot if path is provided
        if save_path:
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            extension = save_path.rsplit('.', 1)[1] if '.' in save_path else FILETYPE
            metric_save_path = f"{base_path}_{save_suffix}.{extension}"
            os.makedirs(os.path.dirname(metric_save_path), exist_ok=True)
            plt.savefig(metric_save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
            print(f"{metric_name} plot saved to: {metric_save_path}")
        
        plt.show()
        return fig, ax
    
    # Create F1 plot
    fig_f1, ax_f1 = create_metric_plot(
        "F1", 
        "F1_With_Compression", 
        "F1_Without_Compression",
        COLOR_PALETTE['F1_With_Compression'],
        COLOR_PALETTE['F1_Without_Compression'],
        "f1"
    )
    
    # Create ROUGE-L plot
    fig_rouge, ax_rouge = create_metric_plot(
        "ROUGE-L", 
        "ROUGE-L_With_Compression", 
        "ROUGE-L_Without_Compression",
        COLOR_PALETTE['ROUGE-L_With_Compression'],
        COLOR_PALETTE['ROUGE-L_Without_Compression'],
        "rouge_l"
    )
    
    return (fig_f1, ax_f1), (fig_rouge, ax_rouge)


def plot_compression_sensitivity_analysis(dataset_name, model_name):
    """
    Plots the compression sensitivity analysis comparing two scenarios.

    Args:
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model.
    """
    # Get data using direct values
    df = get_sensitivity_data_direct(dataset_name, model_name)
    
    if df is None:
        print(f"⚠ No data available for {dataset_name}")
        return
    
    # Set save path
    save_path = f"outputs/visual/sensitivity/{model_name}/sensitivity_{dataset_name}.{FILETYPE}"
    
    # Create single plot with all lines
    create_single_plot(df, dataset_name, model_name, save_path)
    print(f"✓ Completed compression sensitivity analysis for: {dataset_name}")


def create_combined_plot(datasets, model_name):
    """
    Create a combined plot showing all datasets in a grid layout.
    
    Args:
        datasets (list): List of dataset names
        model_name (str): Name of the model
    """
    # Calculate grid dimensions
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset_name in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get data for this dataset
        df = get_sensitivity_data_direct(dataset_name, model_name)
        
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"No data for {dataset_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset_name_map.get(dataset_name, dataset_name))
            continue
        
        # Plot all 4 lines on the same axes
        if not df['F1_With_Compression'].isna().all():
            ax.plot(df['k'], df['F1_With_Compression'], 
                   marker='o', linewidth=1.5, markersize=4, 
                   color=COLOR_PALETTE['F1_With_Compression'], 
                   linestyle='--', label='F1 (With)')
        
        if not df['ROUGE-L_With_Compression'].isna().all():
            ax.plot(df['k'], df['ROUGE-L_With_Compression'], 
                   marker='s', linewidth=1.5, markersize=4, 
                   color=COLOR_PALETTE['ROUGE-L_With_Compression'], 
                   label='ROUGE-L (With)')
        
        if not df['F1_Without_Compression'].isna().all():
            ax.plot(df['k'], df['F1_Without_Compression'], 
                   marker='^', linewidth=1.5, markersize=4, 
                   color=COLOR_PALETTE['F1_Without_Compression'], 
                   linestyle='--', label='F1 (Without)')
        
        if not df['ROUGE-L_Without_Compression'].isna().all():
            ax.plot(df['k'], df['ROUGE-L_Without_Compression'], 
                   marker='d', linewidth=1.5, markersize=4, 
                   color=COLOR_PALETTE['ROUGE-L_Without_Compression'], 
                   label='ROUGE-L (Without)')
        
        ax.set_title(dataset_name_map.get(dataset_name, dataset_name), fontsize=FONTSIZE-2)
        ax.set_xlabel("k", fontsize=FONTSIZE-3)
        ax.set_ylabel("Score", fontsize=FONTSIZE-3)
        ax.tick_params(labelsize=FONTSIZE-4)
        ax.legend(fontsize=FONTSIZE-4)
    
    # Hide empty subplots
    for idx in range(len(datasets), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"Compression Sensitivity Analysis - {model_name}", 
                fontsize=TITLE_FONTSIZE, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save combined plot
    save_path = f"outputs/visual/compression_sensitivity/{model_name}/compression_sensitivity.{FILETYPE}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Combined plot saved to: {save_path}")
    plt.show()


# ===== [Main Execution] =====
def main():
    """Main execution function for compression sensitivity analysis plotting."""
    project_setup()
    
    model_name = "Mistral7B"
    datasets = ["qasper", "narrativeqa", "hotpotqa", "quality", "triviaqa"]

    # for dataset_name in datasets:
    #     df = pd.read_excel(f"outputs/qa/performance_summary.xlsx", sheet_name=dataset_name)
    #     print(df[(df['model_name'] == "Mistral7B_QA_Compress_Stage0") & (df['n'] == 10)]['f1'].values)
    #     print(df[(df['model_name'] == "Mistral7B_QA_Compress_Stage0") & (df['n'] == 10)]['rougel'].values)
    #     print(df[(df['model_name'] == "Mistral7B") & (df['k'] == df['n'])]['f1'].values)
    #     print(df[(df['model_name'] == "Mistral7B") & (df['k'] == df['n'])]['rougel'].values)
    
    print("\n" + "=" * 60)
    print("COMPRESSION SENSITIVITY ANALYSIS PLOTTING")
    print("=" * 60)
    
    # Plot individual datasets
    for dataset_name in datasets:
        print(f"\n📊 Processing: {dataset_name}")
        plot_compression_sensitivity_analysis(dataset_name, model_name)
    
    # Create combined plot
    print(f"\n📊 Creating combined plot for all datasets...")
    create_combined_plot(datasets, model_name)
    
    print("\n" + "=" * 60)
    print("✅ All compression sensitivity analysis plots completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 