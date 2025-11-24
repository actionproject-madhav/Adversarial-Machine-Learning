"""
Publication-Quality Visualizations for Adversarial ML Gap Analysis
Clean, professional figures suitable for academic papers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Professional publication settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Color palette - professional and colorblind-friendly
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D69F24',
    'danger': '#C73E1D',
    'neutral': '#6C757D',
    'light_gray': '#E9ECEF',
}

# Apruzzese baseline
BASELINE = {
    'Attack focus': 72,
    'DL only': 89,
    'No economics': 73,
    'Code released': 51,
    'Real systems': 20,
}

def load_data(csv_path):
    """Load data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers")
    return df

def fig1_yearly_distribution(df, output_dir):
    """
    Figure 1: Publication volume and attack/defense focus by year
    Single clean bar chart with grouped bars
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get counts by year and focus
    focus_year = df.groupby(['Year', 'G1']).size().unstack(fill_value=0)
    
    # Reorder columns: atk, def, both
    if 'both' in focus_year.columns:
        focus_year = focus_year[['atk', 'def', 'both']]
    else:
        focus_year = focus_year[['atk', 'def']]
    
    # Plot grouped bars
    x = np.arange(len(focus_year.index))
    width = 0.25
    
    colors = [COLORS['danger'], COLORS['success'], COLORS['warning']]
    labels = ['Attack', 'Defense', 'Both']
    
    for i, (col, color, label) in enumerate(zip(focus_year.columns, colors, labels)):
        if col in focus_year.columns:
            ax.bar(x + i*width, focus_year[col], width, label=label, 
                   color=color, edgecolor='black', linewidth=0.7)
    
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Number of Papers', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(focus_year.index)
    ax.legend(frameon=True, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_yearly_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_yearly_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_yearly_distribution")

def fig2_gap_score_evolution(df, output_dir):
    """
    Figure 2: Average gap score over time with trend
    Clean line plot showing if gap is narrowing or widening
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate average gap score by year
    gap_by_year = df.groupby('Year')['Traditional_Score'].agg(['mean', 'std'])
    years = gap_by_year.index.values
    means = gap_by_year['mean'].values
    stds = gap_by_year['std'].values
    
    # Plot with error bars
    ax.errorbar(years, means, yerr=stds, marker='o', markersize=8,
                linewidth=2.5, capsize=6, capthick=2, 
                color=COLORS['primary'], ecolor=COLORS['light_gray'],
                label='Mean ± SD')
    
    # Add trend line
    z = np.polyfit(years, means, 1)
    p = np.poly1d(z)
    trend_color = COLORS['success'] if z[0] < 0 else COLORS['danger']
    ax.plot(years, p(years), '--', linewidth=2, color=trend_color,
            label=f'Trend (slope={z[0]:.3f}/year)')
    
    # Reference lines
    ax.axhline(y=2, color=COLORS['success'], linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=4, color=COLORS['danger'], linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Labels
    ax.text(years[0]-0.2, 2, 'Low gap', fontsize=9, color=COLORS['success'], 
            ha='right', va='center')
    ax.text(years[0]-0.2, 4, 'High gap', fontsize=9, color=COLORS['danger'], 
            ha='right', va='center')
    
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Average Gap Score (0-6)', fontweight='bold')
    ax.set_ylim(0, 6)
    ax.legend(frameon=True, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_gap_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_gap_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_gap_evolution")

def fig3_practical_vs_gap(df, output_dir):
    """
    Figure 3: Side-by-side comparison of practical indicators vs gap indicators
    Two panels showing good vs bad practices
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    total = len(df)
    
    # Left panel: Practical indicators (higher is better)
    practical = {
        'Mentions\nEconomics': (df['G5'] == 'YES').sum() / total * 100,
        'Releases\nCode': (df['G6'] == 'YES').sum() / total * 100,
        'Tests Real\nSystems': (df['G7'] == 'YES').sum() / total * 100,
    }
    
    axes[0].bar(practical.keys(), practical.values(), 
                color=COLORS['success'], edgecolor='black', linewidth=0.7, alpha=0.85)
    axes[0].axhline(y=50, color='black', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].set_ylabel('Percentage (%)', fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('(a) Practical Indicators (Higher = Better)', fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (k, v) in enumerate(practical.items()):
        axes[0].text(i, v + 3, f'{v:.1f}%', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10)
    
    # Right panel: Gap indicators (lower is better)
    gap = {
        'Gradient\nBased': df['Flag_Grad'].sum() / total * 100,
        'High Query\nBudget': df['Flag_HighQ'].sum() / total * 100,
        'White-box\nOnly': df['Flag_WB'].sum() / total * 100,
    }
    
    axes[1].bar(gap.keys(), gap.values(), 
                color=COLORS['danger'], edgecolor='black', linewidth=0.7, alpha=0.85)
    axes[1].axhline(y=50, color='black', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_ylabel('Percentage (%)', fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].set_title('(b) Gap Indicators (Lower = Better)', fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (k, v) in enumerate(gap.items()):
        axes[1].text(i, v + 3, f'{v:.1f}%', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_practical_vs_gap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_practical_vs_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_practical_vs_gap")

def fig4_baseline_comparison(df, output_dir):
    """
    Figure 4: Comparison with Apruzzese 2019-2021 baseline
    Clean grouped bar chart showing progress
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total = len(df)
    
    # Prepare data
    metrics = ['Attack\nFocus', 'DL\nOnly', 'No\nEconomics', 'Code\nReleased', 'Real\nSystems']
    baseline_vals = [
        BASELINE['Attack focus'],
        BASELINE['DL only'],
        BASELINE['No economics'],
        BASELINE['Code released'],
        BASELINE['Real systems'],
    ]
    current_vals = [
        (df['G1'] == 'atk').sum() / total * 100,
        (df['G3'] == 'DL').sum() / total * 100,
        df['Flag_NoEcon'].sum() / total * 100,
        (df['G6'] == 'YES').sum() / total * 100,
        (df['G7'] == 'YES').sum() / total * 100,
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='2019-2021 (Apruzzese)',
                   color=COLORS['light_gray'], edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, current_vals, width, label='2022-2025 (This Study)',
                   color=COLORS['primary'], edgecolor='black', linewidth=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True, loc='upper left')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_baseline_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_baseline_comparison")

def fig5_gap_distribution(df, output_dir):
    """
    Figure 5: Gap score distribution histogram
    Clean histogram with color-coded categories
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Count papers by gap score
    gap_counts = df['Traditional_Score'].value_counts().sort_index()
    
    # Color code: green (0-1), orange (2-3), red (4-6)
    colors = []
    for score in gap_counts.index:
        if score < 2:
            colors.append(COLORS['success'])
        elif score < 4:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['danger'])
    
    bars = ax.bar(gap_counts.index, gap_counts.values, color=colors,
                  edgecolor='black', linewidth=0.7, alpha=0.85)
    
    # Add value labels
    for i, (score, count) in enumerate(zip(gap_counts.index, gap_counts.values)):
        ax.text(score, count + 0.5, str(count), ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Gap Score', fontweight='bold')
    ax.set_ylabel('Number of Papers', fontweight='bold')
    ax.set_xticks(range(7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], edgecolor='black', label='Low Gap (0-1)'),
        Patch(facecolor=COLORS['warning'], edgecolor='black', label='Medium Gap (2-3)'),
        Patch(facecolor=COLORS['danger'], edgecolor='black', label='High Gap (4-6)')
    ]
    ax.legend(handles=legend_elements, frameon=True, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_gap_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig5_gap_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig5_gap_distribution")

def fig6_threat_model(df, output_dir):
    """
    Figure 6: Threat model assumptions - model knowledge distribution
    Clean horizontal bar chart
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get T1 distribution
    t1_counts = df['T1'].value_counts()
    if 'NA' in t1_counts.index:
        t1_counts = t1_counts.drop('NA')
    
    total = t1_counts.sum()
    t1_pct = (t1_counts / total * 100).sort_values()
    
    # Color code: green (black-box), orange (gray-box), red (white-box)
    color_map = {
        'Black-box': COLORS['success'],
        'Gray-box': COLORS['warning'],
        'White-box': COLORS['danger']
    }
    colors = [color_map.get(idx, COLORS['neutral']) for idx in t1_pct.index]
    
    bars = ax.barh(t1_pct.index, t1_pct.values, color=colors, 
                   edgecolor='black', linewidth=0.7, alpha=0.85)
    
    # Add value labels
    for i, (idx, val) in enumerate(t1_pct.items()):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Percentage (%)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_threat_model.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig6_threat_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig6_threat_model")

def fig7_attack_data_types(df, output_dir):
    """
    Figure 7: Attack types and data types - two clean pie charts
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Attack types (only for attack papers)
    attack_papers = df[df['G1'].isin(['atk', 'both'])]
    attack_types = attack_papers['G2'].value_counts()
    if 'NA' in attack_types.index:
        attack_types = attack_types.drop('NA')
    
    colors1 = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    wedges1, texts1, autotexts1 = axes[0].pie(
        attack_types.values, labels=attack_types.index,
        autopct='%1.1f%%', startangle=90, colors=colors1,
        textprops={'fontsize': 10, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.7}
    )
    for autotext in autotexts1:
        autotext.set_color('white')
    axes[0].set_title('(a) Attack Types', fontweight='bold', pad=20)
    
    # Right: Data types (all papers)
    data_types = df['G4'].value_counts()
    colors2 = [COLORS['primary'], COLORS['accent'], COLORS['success'], 
               COLORS['danger'], COLORS['neutral']][:len(data_types)]
    
    wedges2, texts2, autotexts2 = axes[1].pie(
        data_types.values, labels=data_types.index,
        autopct='%1.1f%%', startangle=90, colors=colors2,
        textprops={'fontsize': 10, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.7}
    )
    for autotext in autotexts2:
        autotext.set_color('white')
    axes[1].set_title('(b) Data Types', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_attack_data_types.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig7_attack_data_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig7_attack_data_types")

def generate_summary_stats(df, output_dir):
    """Generate clean summary statistics table"""
    total = len(df)
    
    stats = {
        'Metric': [
            'Total Papers',
            'Years Covered',
            'Attack Papers (%)',
            'Defense Papers (%)',
            'Deep Learning Only (%)',
            'Mentions Economics (%)',
            'Releases Code (%)',
            'Tests Real Systems (%)',
            'Gradient-Based (%)',
            'High Query Budget (%)',
            'White-box Assumption (%)',
            'Average Gap Score',
            'High Gap Papers (4-6)',
            'Low Gap Papers (0-1)',
        ],
        'Value': [
            f'{total}',
            f"{df['Year'].min()}-{df['Year'].max()}",
            f"{(df['G1'] == 'atk').sum() / total * 100:.1f}",
            f"{(df['G1'] == 'def').sum() / total * 100:.1f}",
            f"{(df['G3'] == 'DL').sum() / total * 100:.1f}",
            f"{(df['G5'] == 'YES').sum() / total * 100:.1f}",
            f"{(df['G6'] == 'YES').sum() / total * 100:.1f}",
            f"{(df['G7'] == 'YES').sum() / total * 100:.1f}",
            f"{df['Flag_Grad'].sum() / total * 100:.1f}",
            f"{df['Flag_HighQ'].sum() / total * 100:.1f}",
            f"{df['Flag_WB'].sum() / total * 100:.1f}",
            f"{df['Traditional_Score'].mean():.2f}",
            f"{(df['Traditional_Score'] >= 4).sum()} ({(df['Traditional_Score'] >= 4).sum()/total*100:.1f}%)",
            f"{(df['Traditional_Score'] < 2).sum()} ({(df['Traditional_Score'] < 2).sum()/total*100:.1f}%)",
        ]
    }
    
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(df_stats.to_string(index=False))
    print("="*60 + "\n")

def main():
    # Configuration
    CSV_PATH = '/mnt/user-data/uploads/analysis_results_full_papers.csv'
    OUTPUT_DIR = '/mnt/user-data/outputs/figures_clean'
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*60)
    
    # Load data
    df = load_data(CSV_PATH)
    
    # Generate figures
    print("\nGenerating figures...")
    fig1_yearly_distribution(df, OUTPUT_DIR)
    fig2_gap_score_evolution(df, OUTPUT_DIR)
    fig3_practical_vs_gap(df, OUTPUT_DIR)
    fig4_baseline_comparison(df, OUTPUT_DIR)
    fig5_gap_distribution(df, OUTPUT_DIR)
    fig6_threat_model(df, OUTPUT_DIR)
    fig7_attack_data_types(df, OUTPUT_DIR)
    
    # Generate summary
    generate_summary_stats(df, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}/")
    print("Generated: 7 figures (PDF + PNG) + summary CSV")

if __name__ == "__main__":
    main()