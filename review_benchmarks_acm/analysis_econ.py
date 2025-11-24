"""
Research-Practice Gap Analysis - Focused Visualizations
Figures specifically addressing the gap between academic research and industry needs
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

# Color palette
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

def load_data(csv_path):
    """Load data"""
    # Convert Path object to string if needed
    csv_path_str = str(csv_path)
    df = pd.read_csv(csv_path_str)
    print(f"Loaded {len(df)} papers from {csv_path_str}")
    return df

def gap_fig1_six_flags_breakdown(df, output_dir):
    """
    Gap Figure 1: Individual flag analysis showing what contributes to gap
    Horizontal stacked bar showing percentage of papers with each flag
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total = len(df)
    
    # Calculate percentages for each flag
    flags = {
        'Gradient-Based\n(Requires White-box Access)': df['Flag_Grad'].sum() / total * 100,
        'High Query Budget\n(>1000 Queries)': df['Flag_HighQ'].sum() / total * 100,
        'White-box Assumption\n(Unrealistic Threat Model)': df['Flag_WB'].sum() / total * 100,
        'No Economics Discussion\n(Cost/Resource Ignored)': df['Flag_NoEcon'].sum() / total * 100,
        'No Code Released\n(Reproducibility Issue)': df['Flag_NoCode'].sum() / total * 100,
        'No Real System Testing\n(Validation Gap)': df['Flag_NoReal'].sum() / total * 100,
    }
    
    # Sort by percentage
    flags_sorted = dict(sorted(flags.items(), key=lambda x: x[1], reverse=True))
    
    y_pos = np.arange(len(flags_sorted))
    values = list(flags_sorted.values())
    labels = list(flags_sorted.keys())
    
    # Color gradient from red (high) to yellow (medium)
    colors = [COLORS['danger'] if v > 50 else COLORS['warning'] if v > 30 else COLORS['success'] 
              for v in values]
    
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.7, alpha=0.85)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 1.5, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=10)
    
    # Add 50% reference line
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(51, len(flags_sorted)-0.5, '50% threshold', fontsize=9, alpha=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig1_six_flags.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig1_six_flags.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig1_six_flags")

def gap_fig2_flag_evolution(df, output_dir):
    """
    Gap Figure 2: How gap indicators changed over years
    Line plot showing evolution of each flag
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = sorted(df['Year'].unique())
    
    flags = {
        'Gradient-Based': 'Flag_Grad',
        'High Query Budget': 'Flag_HighQ',
        'White-box': 'Flag_WB',
        'No Code': 'Flag_NoCode',
        'No Real Systems': 'Flag_NoReal',
    }
    
    colors_list = [COLORS['danger'], COLORS['warning'], COLORS['primary'], 
                   COLORS['secondary'], COLORS['accent']]
    markers = ['o', 's', '^', 'D', 'v']
    
    for (label, col), color, marker in zip(flags.items(), colors_list, markers):
        percentages = []
        for year in years:
            year_df = df[df['Year'] == year]
            pct = year_df[col].sum() / len(year_df) * 100
            percentages.append(pct)
        
        ax.plot(years, percentages, marker=marker, markersize=8, linewidth=2.5,
                label=label, color=color)
    
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc='best', ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add 50% reference line
    ax.axhline(y=50, color='black', linestyle=':', alpha=0.3, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig2_flag_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig2_flag_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig2_flag_evolution")

def gap_fig3_real_systems_problem(df, output_dir):
    """
    Gap Figure 3: Real system testing - the critical gap
    Emphasize the low percentage of papers testing on real systems
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate by year
    years = sorted(df['Year'].unique())
    real_yes = []
    real_no = []
    
    for year in years:
        year_df = df[df['Year'] == year]
        yes_count = (year_df['G7'] == 'YES').sum()
        no_count = (year_df['G7'] == 'NO').sum()
        total = len(year_df)
        real_yes.append(yes_count / total * 100)
        real_no.append(no_count / total * 100)
    
    x = np.arange(len(years))
    width = 0.6
    
    # Stacked bars
    bars1 = ax.bar(x, real_yes, width, label='Tests Real Systems', 
                   color=COLORS['success'], edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x, real_no, width, bottom=real_yes, label='No Real System Testing',
                   color=COLORS['danger'], edgecolor='black', linewidth=0.7, alpha=0.85)
    
    # Add percentage labels
    for i, (yes, no) in enumerate(zip(real_yes, real_no)):
        # Label for YES
        ax.text(i, yes/2, f'{yes:.1f}%', ha='center', va='center',
                fontweight='bold', fontsize=11, color='white')
        # Label for NO
        ax.text(i, yes + no/2, f'{no:.1f}%', ha='center', va='center',
                fontweight='bold', fontsize=11, color='white')
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig3_real_systems.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig3_real_systems.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig3_real_systems")

def gap_fig4_gradient_query_scatter(df, output_dir):
    """
    Gap Figure 4: Scatter plot showing gradient requirement vs query budget
    Shows clustering of impractical approaches
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create categories for plotting
    # X-axis: Query budget (Low=0, High=1, None=2)
    # Y-axis: Gradient requirement (NO=0, YES=1)
    
    query_map = {'Low': 0, 'High': 1, 'None': 2}
    grad_map = {'NO': 0, 'YES': 1}
    
    df_plot = df.copy()
    df_plot['Query_Numeric'] = df_plot['Q2'].map(query_map)
    df_plot['Grad_Numeric'] = df_plot['Q1'].map(grad_map)
    
    # Add some jitter for visibility
    np.random.seed(42)
    jitter = 0.1
    x = df_plot['Query_Numeric'] + np.random.uniform(-jitter, jitter, len(df_plot))
    y = df_plot['Grad_Numeric'] + np.random.uniform(-jitter, jitter, len(df_plot))
    
    # Color by gap score
    colors = [COLORS['success'] if score < 2 else COLORS['warning'] if score < 4 
              else COLORS['danger'] for score in df_plot['Traditional_Score']]
    
    scatter = ax.scatter(x, y, c=colors, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Query Budget', fontweight='bold')
    ax.set_ylabel('Gradient Requirement', fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Low\n(<1000)', 'High\n(>1000)', 'None\n(No Queries)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No\n(Practical)', 'Yes\n(Impractical)'])
    
    # Add region labels
    ax.text(2, 0, 'Most Practical', ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.3))
    ax.text(1, 1, 'Least Practical', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=COLORS['danger'], alpha=0.3))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='Low Gap (0-1)'),
        Patch(facecolor=COLORS['warning'], label='Medium Gap (2-3)'),
        Patch(facecolor=COLORS['danger'], label='High Gap (4-6)')
    ]
    ax.legend(handles=legend_elements, frameon=True, loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig4_gradient_query.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig4_gradient_query.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig4_gradient_query")

def gap_fig5_practical_score_distribution(df, output_dir):
    """
    Gap Figure 5: Distribution of practical characteristics
    Box plot showing spread of gap scores by different dimensions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create datasets for box plot
    categories = []
    scores = []
    
    # By model knowledge
    for t1 in ['Black-box', 'Gray-box', 'White-box']:
        subset = df[df['T1'] == t1]
        if len(subset) > 0:
            categories.extend([f'{t1}\n(n={len(subset)})'] * len(subset))
            scores.extend(subset['Traditional_Score'].values)
    
    # Separate them
    positions = []
    data_groups = []
    labels = []
    
    current_pos = 1
    for t1 in ['Black-box', 'Gray-box', 'White-box']:
        subset = df[df['T1'] == t1]
        if len(subset) > 0:
            data_groups.append(subset['Traditional_Score'].values)
            positions.append(current_pos)
            labels.append(f'{t1}\n(n={len(subset)})')
            current_pos += 1
    
    # Create box plot
    bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor=COLORS['primary'], alpha=0.7, edgecolor='black'),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    ax.set_ylabel('Gap Score (0-6)', fontweight='bold')
    ax.set_xlabel('Threat Model', fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.5, 6.5)
    
    # Add reference lines
    ax.axhline(y=2, color=COLORS['success'], linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=4, color=COLORS['danger'], linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean'),
    ]
    ax.legend(handles=legend_elements, frameon=True, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig5_score_by_threat.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig5_score_by_threat.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig5_score_by_threat")

def gap_fig6_industry_needs_vs_research(df, output_dir):
    """
    Gap Figure 6: What industry needs vs what research provides
    Side-by-side comparison of industry requirements and research characteristics
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    total = len(df)
    
    # Industry needs (what practitioners want - higher is better)
    industry_needs = {
        'Black-box\nAccess': 100,  # Industry can rarely access model internals
        'Low Query\nBudget': 100,  # Queries are expensive in production
        'No Gradient\nAccess': 100,  # Gradients unavailable in practice
        'Real System\nValidation': 100,  # Must work on actual systems
        'Code\nAvailability': 100,  # Need reproducibility
    }
    
    # Research reality (what papers actually provide)
    research_reality = {
        'Black-box\nAccess': (df['T1'] == 'Black-box').sum() / total * 100,
        'Low Query\nBudget': ((df['Q2'] == 'Low') | (df['Q2'] == 'None')).sum() / total * 100,
        'No Gradient\nAccess': (df['Q1'] == 'NO').sum() / total * 100,
        'Real System\nValidation': (df['G7'] == 'YES').sum() / total * 100,
        'Code\nAvailability': (df['G6'] == 'YES').sum() / total * 100,
    }
    
    labels = list(industry_needs.keys())
    industry_vals = list(industry_needs.values())
    research_vals = [research_reality[k] for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, industry_vals, width, label='Industry Needs',
                   color=COLORS['success'], edgecolor='black', linewidth=0.7, alpha=0.4)
    bars2 = ax.bar(x + width/2, research_vals, width, label='Research Provides',
                   color=COLORS['primary'], edgecolor='black', linewidth=0.7)
    
    # Add value labels
    for bars in [bars2]:  # Only label research bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add gap arrows for large differences
    for i, (ind, res) in enumerate(zip(industry_vals, research_vals)):
        gap = ind - res
        if gap > 20:  # Show arrow if gap > 20%
            y_start = res + 5
            y_end = ind - 5
            ax.annotate('', xy=(i, y_end), xytext=(i, y_start),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(i + 0.15, (y_start + y_end) / 2, f'{gap:.0f}%\ngap',
                   fontsize=8, color='red', fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig6_industry_vs_research.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig6_industry_vs_research.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig6_industry_vs_research")

def gap_fig7_exemplary_papers(df, output_dir):
    """
    Gap Figure 7: Characteristics of low-gap papers (exemplars)
    Compare papers with gap score 0-1 vs 4-6
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Split into low and high gap papers
    low_gap = df[df['Traditional_Score'] < 2]
    high_gap = df[df['Traditional_Score'] >= 4]
    
    if len(low_gap) == 0 or len(high_gap) == 0:
        print("Warning: Not enough papers in gap categories")
        return
    
    # Compare characteristics
    characteristics = {
        'Black-box\nAccess': [
            (low_gap['T1'] == 'Black-box').sum() / len(low_gap) * 100,
            (high_gap['T1'] == 'Black-box').sum() / len(high_gap) * 100
        ],
        'Releases\nCode': [
            (low_gap['G6'] == 'YES').sum() / len(low_gap) * 100,
            (high_gap['G6'] == 'YES').sum() / len(high_gap) * 100
        ],
        'Tests Real\nSystems': [
            (low_gap['G7'] == 'YES').sum() / len(low_gap) * 100,
            (high_gap['G7'] == 'YES').sum() / len(high_gap) * 100
        ],
        'Mentions\nEconomics': [
            (low_gap['G5'] == 'YES').sum() / len(low_gap) * 100,
            (high_gap['G5'] == 'YES').sum() / len(high_gap) * 100
        ],
        'No Gradient\nNeeded': [
            (low_gap['Q1'] == 'NO').sum() / len(low_gap) * 100,
            (high_gap['Q1'] == 'NO').sum() / len(high_gap) * 100
        ],
    }
    
    labels = list(characteristics.keys())
    low_vals = [characteristics[k][0] for k in labels]
    high_vals = [characteristics[k][1] for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, low_vals, width, 
                   label=f'Low Gap Papers (n={len(low_gap)})',
                   color=COLORS['success'], edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, high_vals, width, 
                   label=f'High Gap Papers (n={len(high_gap)})',
                   color=COLORS['danger'], edgecolor='black', linewidth=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig7_exemplary_papers.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig7_exemplary_papers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig7_exemplary_papers")

def gap_fig8_cumulative_gap(df, output_dir):
    """
    Gap Figure 8: Cumulative distribution showing proportion of papers at each gap level
    CDF-style plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate cumulative distribution
    gap_scores = df['Traditional_Score'].value_counts().sort_index()
    cumulative = gap_scores.cumsum() / len(df) * 100
    
    # Plot as step function
    scores = list(cumulative.index)
    cum_pct = list(cumulative.values)
    
    # Extend to show full range
    scores_extended = [-0.5] + scores + [6.5]
    cum_pct_extended = [0] + cum_pct + [100]
    
    ax.step(scores_extended, cum_pct_extended, where='post', linewidth=3,
            color=COLORS['primary'])
    ax.fill_between(scores_extended, cum_pct_extended, alpha=0.3, 
                     step='post', color=COLORS['primary'])
    
    # Add vertical lines at category boundaries
    ax.axvline(x=2, color=COLORS['success'], linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=4, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    
    # Add labels for regions
    ax.text(0.5, 50, 'Low Gap', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.3))
    ax.text(3, 50, 'Medium Gap', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.3))
    ax.text(5, 50, 'High Gap', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['danger'], alpha=0.3))
    
    # Add percentage labels at key points
    for score in [1, 3, 5]:
        if score in cumulative.index:
            pct = cumulative[score]
            ax.plot(score, pct, 'o', markersize=10, color='red', zorder=5)
            ax.text(score + 0.2, pct, f'{pct:.0f}%', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Gap Score', fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_fig8_cumulative.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gap_fig8_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: gap_fig8_cumulative")

def main():
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    
    # Configuration - use relative paths
    CSV_PATH = script_dir / 'analysis_results_full_papers.csv'
    # Fallback to other CSV files if the primary one doesn't exist
    if not CSV_PATH.exists():
        CSV_PATH = script_dir / 'analysis_results.csv'
    
    OUTPUT_DIR = script_dir / 'outputs' / 'gap_figures'
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RESEARCH-PRACTICE GAP FOCUSED VISUALIZATIONS")
    print("="*60)
    
    # Load data
    df = load_data(CSV_PATH)
    
    # Generate gap-focused figures
    print("\nGenerating gap analysis figures...")
    gap_fig1_six_flags_breakdown(df, OUTPUT_DIR)
    gap_fig2_flag_evolution(df, OUTPUT_DIR)
    gap_fig3_real_systems_problem(df, OUTPUT_DIR)
    gap_fig4_gradient_query_scatter(df, OUTPUT_DIR)
    gap_fig5_practical_score_distribution(df, OUTPUT_DIR)
    gap_fig6_industry_needs_vs_research(df, OUTPUT_DIR)
    gap_fig7_exemplary_papers(df, OUTPUT_DIR)
    gap_fig8_cumulative_gap(df, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}/")
    print("Generated: 8 gap-focused figures (PDF + PNG)")
    print("\nFigure descriptions:")
    print("1. Six flags breakdown - What creates the gap")
    print("2. Flag evolution - Gap trends over time")
    print("3. Real systems problem - Critical validation gap")
    print("4. Gradient-query scatter - Impractical combinations")
    print("5. Score by threat model - Gap varies by assumptions")
    print("6. Industry needs vs research - Direct comparison")
    print("7. Exemplary papers - What makes low-gap research")
    print("8. Cumulative distribution - Overall gap landscape")

if __name__ == "__main__":
    main()