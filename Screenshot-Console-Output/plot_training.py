#!/usr/bin/env python3
"""
Training visualization script for DQN Stock Trading Agent.

Reads training_log.csv (exported from C# TrainingLoop) and generates:
1. TD Loss vs Episode
2. Cumulative Return vs Episode
3. Portfolio Value over Test Period (placeholder - reads test_portfolio.csv if exists)
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns


# Set professional style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'


# Color palette (brand-neutral, accessible)
COLORS = {
    'primary': '#2C5F8A',      # Deep blue
    'secondary': '#E86C00',    # Orange
    'accent': '#3A923A',       # Green
    'background': '#F8F9FA',
    'grid': '#D0D5DD',
    'text': '#1D2939',
}


def load_training_log(csv_path: str) -> pd.DataFrame:
    """Load and validate training_log.csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training log not found at: {csv_path}\n"
            f"Run the C# training with TrainingLoop.ExportToCsv() to generate it."
        )

    df = pd.read_csv(csv_path)

    required_cols = ['Episode', 'TD_Loss', 'Cumulative_Return', 'Epsilon']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    return df


def load_test_portfolio(csv_path: str) -> pd.DataFrame | None:
    """Load test_portfolio.csv if it exists."""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if 'Step' not in df.columns or 'Portfolio_Value' not in df.columns:
        print(f"Warning: test_portfolio.csv missing required columns (Step, Portfolio_Value). Skipping portfolio plot.")
        return None

    return df


def plot_td_loss(df: pd.DataFrame, output_path: str):
    """Plot TD Loss vs Episode."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Episode'], df['TD_Loss'],
            color=COLORS['primary'], linewidth=1.5, label='TD Loss')

    ax.set_xlabel('Episode', fontsize=12, fontweight='medium')
    ax.set_ylabel('TD Loss', fontsize=12, fontweight='medium')
    ax.set_title('TD Loss vs Episode', fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cumulative_return(df: pd.DataFrame, output_path: str):
    """Plot Cumulative Return vs Episode (as percentage)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    returns_pct = df['Cumulative_Return'] * 100

    ax.plot(df['Episode'], returns_pct,
            color=COLORS['accent'], linewidth=1.5, label='Cumulative Return')

    ax.set_xlabel('Episode', fontsize=12, fontweight='medium')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='medium')
    ax.set_title('Cumulative Return vs Episode', fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_portfolio_value(test_df: pd.DataFrame | None, output_path: str):
    """Plot Portfolio Value over Test Period (placeholder)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if test_df is None:
        # Placeholder: create empty plot with informative message
        ax.text(0.5, 0.5,
                'Portfolio Value over Test Period\n\n'
                'Run evaluation with TrainingLoop.Evaluate() and export\n'
                'test_portfolio.csv with columns: Step, Portfolio_Value',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color=COLORS['text'], alpha=0.6,
                bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['background'],
                          edgecolor=COLORS['grid'], alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(test_df['Step'], test_df['Portfolio_Value'],
                color=COLORS['secondary'], linewidth=1.5, label='Portfolio Value')

        ax.set_xlabel('Time Step', fontsize=12, fontweight='medium')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='medium')
        ax.set_title('Portfolio Value over Test Period', fontsize=14, fontweight='bold', pad=15)

        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_log_path = os.path.join(script_dir, 'training_log.csv')
    test_portfolio_path = os.path.join(script_dir, 'test_portfolio.csv')

    output_dir = script_dir
    td_loss_path = os.path.join(output_dir, 'td_loss.png')
    cum_return_path = os.path.join(output_dir, 'cumulative_return.png')
    portfolio_path = os.path.join(output_dir, 'portfolio_value.png')

    try:
        # Load training data
        print(f"Loading training log from: {training_log_path}")
        df = load_training_log(training_log_path)
        print(f"Loaded {len(df)} episodes")

        # Load test portfolio (optional)
        test_df = load_test_portfolio(test_portfolio_path)
        if test_df is not None:
            print(f"Loaded test portfolio with {len(test_df)} steps")
        else:
            print("No test_portfolio.csv found - creating placeholder for portfolio plot")

        # Generate plots
        plot_td_loss(df, td_loss_path)
        plot_cumulative_return(df, cum_return_path)
        plot_portfolio_value(test_df, portfolio_path)

        print("\nAll plots generated successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()