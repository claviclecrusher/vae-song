#!/usr/bin/env python3
"""
Plotting script for experimental results.
Reads CSV files with exp_lip_ prefix and creates line plots showing KL vs L(z) trade-offs.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def parse_experiment_name(filename):
    """Extract experiment name from filename like 'exp_lip_lrvae_linear_two_imgauss.csv'"""
    basename = os.path.basename(filename)
    if basename.startswith('exp_lip_'):
        return basename[8:-4]  # Remove 'exp_lip_' prefix and '.csv' suffix
    return basename[:-4]

def load_and_combine_data(input_dir):
    """Load all CSV files with exp_lip_ prefix and combine them"""
    csv_files = glob.glob(os.path.join(input_dir, "exp_lip_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found with 'exp_lip_' prefix in {input_dir}")
        return None
    
    combined_data = []
    experiment_names = []
    
    for csv_file in csv_files:
        exp_name = parse_experiment_name(csv_file)
        experiment_names.append(exp_name)
        
        try:
            df = pd.read_csv(csv_file)
            df['experiment'] = exp_name
            combined_data.append(df)
            print(f"Loaded {csv_file} -> experiment: {exp_name}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not combined_data:
        return None, []
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df, experiment_names

def select_best_run(df, selection_method='kl_min'):
    """Select the best run for each (alpha, beta) combination"""
    if selection_method == 'kl_min':
        # Select run with minimum KL for each (alpha, beta) combination
        return df.loc[df.groupby(['alpha', 'beta'])['kl'].idxmin()]
    elif selection_method == 'lipschitz_min':
        # Select run with minimum L(z) for each (alpha, beta) combination
        return df.loc[df.groupby(['alpha', 'beta'])['L(z)'].idxmin()]
    elif selection_method == 'kl_max':
        # Select run with maximum KL for each (alpha, beta) combination
        return df.loc[df.groupby(['alpha', 'beta'])['kl'].idxmax()]
    elif selection_method == 'lipschitz_max':
        # Select run with maximum L(z) for each (alpha, beta) combination
        return df.loc[df.groupby(['alpha', 'beta'])['L(z)'].idxmax()]
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

def create_plot(df, output_dir, experiment_name):
    """Create subplot with KL and L(z) vs beta for different alpha values"""
    
    # Text size multiplier for easy adjustment
    text_scale = 2.2
    
    # Get unique alpha values and sort them
    alpha_values = sorted(df['alpha'].unique())
    beta_values = sorted(df['beta'].unique())
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color map for different alpha values using viridis palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
    
    # Plot for each alpha value
    for i, alpha in enumerate(alpha_values):
        alpha_data = df[df['alpha'] == alpha].sort_values('beta')
        
        if len(alpha_data) == 0:
            continue
            
        # Use red color for alpha=0, viridis for others
        if alpha == 0.0:
            color = '#CC0000'
            label = f'α={alpha} (β-VAE)'
        else:
            color = colors[i]
            label = f'α={alpha} (Ours)'
        
        # Plot KL (left subplot)
        ax1.plot(alpha_data['beta'], alpha_data['kl'], 
                '--s', color=color, linewidth=4, markersize=14, 
                label=label)
        
        # Plot L(z) (right subplot)
        ax2.plot(alpha_data['beta'], alpha_data['L(z)'], 
                '-o', color=color, linewidth=4, markersize=14, 
                label=label)
    
    # Customize left subplot (KL)
    ax1.set_xlabel('β (Regularization Weight)', fontsize=14 * text_scale)
    #ax1.set_ylabel('Mean KLD', fontsize=14 * text_scale, labelpad=-50)
    # Manually position Y-axis label higher
    ax1.text(-0.05, 0.78, 'Mean KLD', transform=ax1.transAxes, fontsize=14 * text_scale, 
             rotation=90, ha='center', va='top')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('KL Divergence with β', fontsize=16 * text_scale)
    ax1.legend(fontsize=10 * text_scale)
    ax1.set_xticks(beta_values)
    ax1.tick_params(axis='both', which='major', labelsize=14 * text_scale)
    ax1.tick_params(axis='both', which='minor', labelsize=12 * text_scale)
    
    # Customize right subplot (L(z))
    ax2.set_xlabel('β (Regularization Weight)', fontsize=14 * text_scale)
    #ax2.set_ylabel('Mean L(z)', fontsize=14 * text_scale, labelpad=-50)
    # Manually position Y-axis label higher
    ax2.text(-0.05, 0.72, 'Mean L(z)', transform=ax2.transAxes, fontsize=14 * text_scale, 
             rotation=90, ha='center', va='top')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Local bi-Lipschitz with β', fontsize=16 * text_scale)
    ax2.legend(fontsize=10 * text_scale, loc='center right', bbox_to_anchor=(0.98, 0.55))
    ax2.set_xticks(beta_values)
    ax2.tick_params(axis='both', which='major', labelsize=14 * text_scale)
    ax2.tick_params(axis='both', which='minor', labelsize=12 * text_scale)
    
    # Adjust layout with more space between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.16)
    
    # Save combined plot
    output_file = os.path.join(output_dir, f"{experiment_name}_plot.svg")
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot experimental results from CSV files')
    parser.add_argument('--input_dir', type=str, default='input_data',
                       help='Input directory containing CSV files (default: input_data)')
    parser.add_argument('--output_dir', type=str, default='output_figure',
                       help='Output directory for plots (default: output_figure)')
    parser.add_argument('--selection_method', type=str, default='kl_min',
                       choices=['kl_min', 'kl_max', 'lipschitz_min', 'lipschitz_max'],
                       help='Method to select best run for each (alpha, beta) combination (default: kl_min)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment name to plot (default: plot all experiments)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and combine data
    print("Loading data...")
    combined_df, experiment_names = load_and_combine_data(args.input_dir)
    
    if combined_df is None:
        print("No data found!")
        return
    
    print(f"Found experiments: {experiment_names}")
    print(f"Total data points: {len(combined_df)}")
    
    # Filter by specific experiment if requested
    if args.experiment:
        if args.experiment not in experiment_names:
            print(f"Experiment '{args.experiment}' not found. Available: {experiment_names}")
            return
        combined_df = combined_df[combined_df['experiment'] == args.experiment]
        experiment_names = [args.experiment]
    
    # Process each experiment
    for exp_name in experiment_names:
        print(f"\nProcessing experiment: {exp_name}")
        
        # Filter data for this experiment
        exp_data = combined_df[combined_df['experiment'] == exp_name].copy()
        
        if len(exp_data) == 0:
            print(f"No data found for experiment: {exp_name}")
            continue
        
        # Handle infinite values
        exp_data = exp_data.replace([np.inf, -np.inf], np.nan)
        exp_data = exp_data.dropna()
        
        if len(exp_data) == 0:
            print(f"No valid data after cleaning for experiment: {exp_name}")
            continue
        
        # Select best run for each (alpha, beta) combination
        selected_data = select_best_run(exp_data, args.selection_method)
        
        print(f"Selected {len(selected_data)} data points using method: {args.selection_method}")
        print(f"Alpha values: {sorted(selected_data['alpha'].unique())}")
        print(f"Beta values: {sorted(selected_data['beta'].unique())}")
        
        # Create plot
        create_plot(selected_data, args.output_dir, exp_name)
    
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
