"""
Compare metrics between proprietary API models and local self-hosted models.

This script helps address reviewer questions about real-world cost and latency comparisons.
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


from src.utils.data_utils import read_jsonl
from src.utils.utility import print_colored


def load_results(file_path: str) -> List[Dict]:
    """Load results from JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    return read_jsonl(file_path)


def calculate_proprietary_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for proprietary model results."""
    metrics = {
        'num_samples': len(results),
        'total_input_tokens': sum(r.get('input_tokens', 0) for r in results),
        'total_output_tokens': sum(r.get('output_tokens', 0) for r in results),
        'total_tokens': sum(r.get('total_tokens', 0) for r in results),
        'total_cost_usd': sum(r.get('cost_usd', 0) for r in results),
        'total_latency': sum(r.get('latency_seconds', 0) for r in results),
    }

    if metrics['num_samples'] > 0:
        metrics['avg_input_tokens'] = metrics['total_input_tokens'] / metrics['num_samples']
        metrics['avg_output_tokens'] = metrics['total_output_tokens'] / metrics['num_samples']
        metrics['avg_total_tokens'] = metrics['total_tokens'] / metrics['num_samples']
        metrics['avg_cost_usd'] = metrics['total_cost_usd'] / metrics['num_samples']
        metrics['avg_latency'] = metrics['total_latency'] / metrics['num_samples']
        metrics['tokens_per_second'] = metrics['total_output_tokens'] / metrics['total_latency'] if metrics['total_latency'] > 0 else 0

    return metrics


def calculate_local_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for local model results."""
    # Filter results that have metrics
    results_with_metrics = [r for r in results if 'input_tokens' in r or 'total_tokens' in r]

    if not results_with_metrics:
        print_colored("Warning: No metrics found in local model results. Did you run with --track_metrics?", "yellow")
        return {
            'num_samples': len(results),
            'has_metrics': False,
        }

    metrics = {
        'num_samples': len(results_with_metrics),
        'has_metrics': True,
        'total_input_tokens': sum(r.get('input_tokens', 0) for r in results_with_metrics),
        'total_output_tokens': sum(r.get('output_tokens', 0) for r in results_with_metrics),
        'total_tokens': sum(r.get('total_tokens', 0) for r in results_with_metrics),
        'total_latency': sum(r.get('e2e_latency_seconds', 0) for r in results_with_metrics),
        'avg_gpu_memory_peak_mb': np.mean([r.get('gpu_memory_peak_mb', 0) for r in results_with_metrics]),
    }

    if metrics['num_samples'] > 0:
        metrics['avg_input_tokens'] = metrics['total_input_tokens'] / metrics['num_samples']
        metrics['avg_output_tokens'] = metrics['total_output_tokens'] / metrics['num_samples']
        metrics['avg_total_tokens'] = metrics['total_tokens'] / metrics['num_samples']
        metrics['avg_latency'] = metrics['total_latency'] / metrics['num_samples']
        metrics['tokens_per_second'] = metrics['total_output_tokens'] / metrics['total_latency'] if metrics['total_latency'] > 0 else 0

    return metrics


def estimate_local_model_cost(metrics: Dict, gpu_type: str = "A100") -> Dict:
    """
    Estimate cost for running local model on cloud GPU.

    GPU hourly rates (approximate):
    - A100 (40GB): $2.00/hour
    - A100 (80GB): $3.00/hour
    - H100: $4.00/hour
    - L4: $0.60/hour
    - T4: $0.35/hour
    """
    gpu_hourly_cost = {
        "A100-40GB": 2.00,
        "A100-80GB": 3.00,
        "H100": 4.00,
        "L4": 0.60,
        "T4": 0.35,
    }

    hourly_cost = gpu_hourly_cost.get(gpu_type, 2.00)
    total_hours = metrics.get('total_latency', 0) / 3600  # Convert seconds to hours

    return {
        'gpu_type': gpu_type,
        'hourly_cost_usd': hourly_cost,
        'total_gpu_hours': total_hours,
        'total_cost_usd': total_hours * hourly_cost,
        'avg_cost_per_query_usd': (total_hours * hourly_cost) / metrics.get('num_samples', 1) if metrics.get('num_samples', 0) > 0 else 0,
    }


def print_comparison(proprietary_metrics: Dict, local_metrics: Dict, local_cost_estimate: Dict, model_names: Dict):
    """Print detailed comparison between proprietary and local models."""

    print_colored("\n" + "=" * 80, "blue")
    print_colored("Model Comparison: Proprietary vs Local", "blue")
    print_colored("=" * 80 + "\n", "blue")

    # Model information
    print_colored("Models:", "white")
    print(f"  Proprietary: {model_names['proprietary']}")
    print(f"  Local:       {model_names['local']}")
    print()

    # Sample counts
    print_colored("Sample Counts:", "white")
    print(f"  Proprietary: {proprietary_metrics['num_samples']:,} samples")
    print(f"  Local:       {local_metrics['num_samples']:,} samples")
    print()

    if not local_metrics.get('has_metrics', True):
        print_colored("⚠ Local model results don't have metrics. Run with --track_metrics flag!", "yellow")
        return

    # Token usage
    print_colored("Token Usage (per query):", "white")
    print(f"  {'Metric':<25} {'Proprietary':<20} {'Local':<20} {'Difference':<20}")
    print(f"  {'-' * 25} {'-' * 20} {'-' * 20} {'-' * 20}")
    print(f"  {'Input tokens':<25} {proprietary_metrics['avg_input_tokens']:>15.1f}    {local_metrics['avg_input_tokens']:>15.1f}    {local_metrics['avg_input_tokens'] - proprietary_metrics['avg_input_tokens']:>+15.1f}")
    print(f"  {'Output tokens':<25} {proprietary_metrics['avg_output_tokens']:>15.1f}    {local_metrics['avg_output_tokens']:>15.1f}    {local_metrics['avg_output_tokens'] - proprietary_metrics['avg_output_tokens']:>+15.1f}")
    print(f"  {'Total tokens':<25} {proprietary_metrics['avg_total_tokens']:>15.1f}    {local_metrics['avg_total_tokens']:>15.1f}    {local_metrics['avg_total_tokens'] - proprietary_metrics['avg_total_tokens']:>+15.1f}")
    print()

    # Latency
    print_colored("Latency (per query):", "white")
    print(f"  {'Metric':<25} {'Proprietary':<20} {'Local':<20} {'Speedup':<20}")
    print(f"  {'-' * 25} {'-' * 20} {'-' * 20} {'-' * 20}")
    print(f"  {'Avg latency (seconds)':<25} {proprietary_metrics['avg_latency']:>15.3f}    {local_metrics['avg_latency']:>15.3f}    {proprietary_metrics['avg_latency'] / local_metrics['avg_latency']:>15.2f}x")
    print(f"  {'Tokens/second':<25} {proprietary_metrics['tokens_per_second']:>15.1f}    {local_metrics['tokens_per_second']:>15.1f}    {local_metrics['tokens_per_second'] / proprietary_metrics['tokens_per_second']:>15.2f}x")
    print()

    # Cost comparison
    print_colored("Cost Comparison:", "white")
    print(f"  Proprietary API ({model_names['proprietary']}):")
    print(f"    Total cost:     ${proprietary_metrics['total_cost_usd']:.4f}")
    print(f"    Cost per query: ${proprietary_metrics['avg_cost_usd']:.6f}")
    print()
    print(f"  Local GPU ({local_cost_estimate['gpu_type']}):")
    print(f"    GPU hourly cost: ${local_cost_estimate['hourly_cost_usd']:.2f}/hour")
    print(f"    Total GPU hours: {local_cost_estimate['total_gpu_hours']:.4f}")
    print(f"    Total cost:      ${local_cost_estimate['total_cost_usd']:.4f}")
    print(f"    Cost per query:  ${local_cost_estimate['avg_cost_per_query_usd']:.6f}")
    print()

    # Cost savings
    if local_cost_estimate['total_cost_usd'] < proprietary_metrics['total_cost_usd']:
        savings = proprietary_metrics['total_cost_usd'] - local_cost_estimate['total_cost_usd']
        savings_pct = (savings / proprietary_metrics['total_cost_usd']) * 100
        print_colored(f"  💰 Local GPU saves ${savings:.4f} ({savings_pct:.1f}% cheaper)", "green")
    else:
        extra_cost = local_cost_estimate['total_cost_usd'] - proprietary_metrics['total_cost_usd']
        extra_cost_pct = (extra_cost / proprietary_metrics['total_cost_usd']) * 100
        print_colored(f"  💰 Proprietary API saves ${extra_cost:.4f} ({extra_cost_pct:.1f}% cheaper)", "yellow")

    print()

    # Memory usage for local model
    if local_metrics.get('avg_gpu_memory_peak_mb', 0) > 0:
        print_colored("GPU Memory (Local Model):", "white")
        print(f"  Peak GPU memory: {local_metrics['avg_gpu_memory_peak_mb']:.1f} MB ({local_metrics['avg_gpu_memory_peak_mb'] / 1024:.2f} GB)")
        print()

    print_colored("=" * 80 + "\n", "blue")


def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare proprietary vs local model metrics")
    parser.add_argument("--proprietary", required=True, help="Path to proprietary model results JSONL")
    parser.add_argument("--local", required=True, help="Path to local model results JSONL")
    parser.add_argument("--gpu-type", default="A100-40GB", choices=["A100-40GB", "A100-80GB", "H100", "L4", "T4"],
                        help="GPU type for cost estimation")
    parser.add_argument("--proprietary-name", default="GPT-4o", help="Proprietary model name for display")
    parser.add_argument("--local-name", default="Local Model", help="Local model name for display")
    parser.add_argument("--output-csv", help="Optional: Save comparison to CSV file")

    args = parser.parse_args()

    # Load results
    print_colored("Loading results...", "cyan")
    proprietary_results = load_results(args.proprietary)
    local_results = load_results(args.local)

    # Calculate metrics
    proprietary_metrics = calculate_proprietary_metrics(proprietary_results)
    local_metrics = calculate_local_metrics(local_results)
    local_cost_estimate = estimate_local_model_cost(local_metrics, args.gpu_type)

    # Print comparison
    model_names = {
        'proprietary': args.proprietary_name,
        'local': args.local_name,
    }
    print_comparison(proprietary_metrics, local_metrics, local_cost_estimate, model_names)

    # Save to CSV if requested
    if args.output_csv:
        comparison_data = {
            'Model': [args.proprietary_name, args.local_name],
            'Samples': [proprietary_metrics['num_samples'], local_metrics['num_samples']],
            'Avg Input Tokens': [proprietary_metrics.get('avg_input_tokens', 0), local_metrics.get('avg_input_tokens', 0)],
            'Avg Output Tokens': [proprietary_metrics.get('avg_output_tokens', 0), local_metrics.get('avg_output_tokens', 0)],
            'Avg Total Tokens': [proprietary_metrics.get('avg_total_tokens', 0), local_metrics.get('avg_total_tokens', 0)],
            'Avg Latency (s)': [proprietary_metrics.get('avg_latency', 0), local_metrics.get('avg_latency', 0)],
            'Tokens/Second': [proprietary_metrics.get('tokens_per_second', 0), local_metrics.get('tokens_per_second', 0)],
            'Total Cost (USD)': [proprietary_metrics.get('total_cost_usd', 0), local_cost_estimate.get('total_cost_usd', 0)],
            'Cost per Query (USD)': [proprietary_metrics.get('avg_cost_usd', 0), local_cost_estimate.get('avg_cost_per_query_usd', 0)],
        }
        df = pd.DataFrame(comparison_data)
        df.to_csv(args.output_csv, index=False)
        print_colored(f"✓ Comparison saved to {args.output_csv}", "green")


if __name__ == "__main__":
    main()
