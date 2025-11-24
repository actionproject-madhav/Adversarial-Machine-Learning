"""
Adversarial ML Research-Practice Gap Analysis - Visualization Script
Generates publication-quality charts showing trends and gaps
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Apruzzese 2019-2021 baseline data
APRUZZESE_BASELINE = {
    'Attack focus': 72,
    'DL only': 89,
    'No economics': 73,
    'Code released': 51,
    'Real systems': 20,
    'Gradient-based': 65,  # Estimated
    'White-box': 55,  # Estimated
}

def load_data(csv_path):
    """Load and validate the analysis results"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers from {csv_path}")
    print(f"Years: {sorted(df['Year'].unique())}")
    return df

def compute_statistics(df):
    """Compute all statistics needed for visualizations"""
    total = len(df)
    
    stats = {
        'total_papers': total,
        'by_year': df['Year'].value_counts().sort_index(),
        
        # Basic characteristics
        'attack_pct': (df['G1'] == 'atk').sum() / total * 100,
        'defense_pct': (df['G1'] == 'def').sum() / total * 100,
        'both_pct': (df['G1'] == 'both').sum() / total * 100,
        'dl_only_pct': (df['G3'] == 'DL').sum() / total * 100,
        'traditional_pct': (df['G3'] == 'Traditional').sum() / total * 100,
        'both_ml_pct': (df['G3'] == 'Both').sum() / total * 100,
        
        # Attack types
        'evasion_pct': (df['G2'] == 'Evasion').sum() / total * 100,
        'poisoning_pct': (df['G2'] == 'Poisoning').sum() / total * 100,
        'privacy_pct': (df['G2'] == 'Privacy').sum() / total * 100,
        'multiple_pct': (df['G2'] == 'Multiple').sum() / total * 100,
        
        # Data types
        'images_pct': (df['G4'] == 'Images').sum() / total * 100,
        'text_pct': (df['G4'] == 'Text').sum() / total * 100,
        'audio_pct': (df['G4'] == 'Audio').sum() / total * 100,
        'malware_pct': (df['G4'] == 'Malware').sum() / total * 100,
        'other_pct': (df['G4'] == 'Other').sum() / total * 100,
        
        # Practical indicators (GOOD)
        'econ_yes_pct': (df['G5'] == 'YES').sum() / total * 100,
        'code_yes_pct': (df['G6'] == 'YES').sum() / total * 100,
        'real_yes_pct': (df['G7'] == 'YES').sum() / total * 100,
        
        # Academic indicators (GAP)
        'grad_pct': df['Flag_Grad'].sum() / total * 100,
        'high_query_pct': df['Flag_HighQ'].sum() / total * 100,
        'white_box_pct': df['Flag_WB'].sum() / total * 100,
        'no_econ_pct': df['Flag_NoEcon'].sum() / total * 100,
        'no_code_pct': df['Flag_NoCode'].sum() / total * 100,
        'no_real_pct': df['Flag_NoReal'].sum() / total * 100,
        
        # Threat model
        'white_box_t1': (df['T1'] == 'White-box').sum() / total * 100,
        'gray_box_t1': (df['T1'] == 'Gray-box').sum() / total * 100,
        'black_box_t1': (df['T1'] == 'Black-box').sum() / total * 100,
        
        # Gap scores
        'avg_gap_score': df['Traditional_Score'].mean(),
        'high_gap': (df['Traditional_Score'] >= 4).sum(),
        'med_gap': ((df['Traditional_Score'] >= 2) & (df['Traditional_Score'] < 4)).sum(),
        'low_gap': (df['Traditional_Score'] < 2).sum(),
    }
    
    return stats

def plot_yearly_trends(df, output_dir):
    """Figure 1: Yearly publication trends and focus distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Papers per year
    year_counts = df['Year'].value_counts().sort_index()
    axes[0].bar(year_counts.index, year_counts.values, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Number of Papers')
    axes[0].set_title('(a) Papers Published per Year')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(year_counts.values):
        axes[0].text(year_counts.index[i], v + 0.5, str(v), ha='center', va='bottom')
    
    # Right: Focus distribution by year
    focus_by_year = df.groupby(['Year', 'G1']).size().unstack(fill_value=0)
    focus_by_year_pct = focus_by_year.div(focus_by_year.sum(axis=1), axis=0) * 100
    
    focus_by_year_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                           color=['#d62728', '#2ca02c', '#ff7f0e'], 
                           edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('(b) Research Focus Distribution by Year')
    axes[1].legend(title='Focus', labels=['Attack', 'Defense', 'Both'])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_yearly_trends.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_yearly_trends.pdf', bbox_inches='tight')
    print("Generated: fig1_yearly_trends.png/pdf")
    plt.close()

def plot_attack_types(df, output_dir):
    """Figure 2: Attack type distribution"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Filter only attack papers
    attack_papers = df[df['G1'].isin(['atk', 'both'])]
    attack_types = attack_papers['G2'].value_counts()
    
    # Remove NA
    attack_types = attack_types[attack_types.index != 'NA']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    wedges, texts, autotexts = ax.pie(attack_types.values, labels=attack_types.index, 
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors, textprops={'fontsize': 10})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax.set_title('Attack Type Distribution (n={})'.format(len(attack_papers)))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_attack_types.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_attack_types.pdf', bbox_inches='tight')
    print("Generated: fig2_attack_types.png/pdf")
    plt.close()

def plot_data_types(df, output_dir):
    """Figure 3: Data type distribution by year"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Data types by year
    data_by_year = df.groupby(['Year', 'G4']).size().unstack(fill_value=0)
    
    # Calculate percentages
    data_by_year_pct = data_by_year.div(data_by_year.sum(axis=1), axis=0) * 100
    
    # Plot grouped bar chart
    data_by_year_pct.plot(kind='bar', ax=ax, width=0.8, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                          edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Data Type Distribution by Year')
    ax.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_data_types.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_data_types.pdf', bbox_inches='tight')
    print("Generated: fig3_data_types.png/pdf")
    plt.close()

def plot_practical_indicators(df, stats, output_dir):
    """Figure 4: Practical indicators (good practices)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Compare by year
    years = sorted(df['Year'].unique())
    metrics = ['G5', 'G6', 'G7']
    metric_labels = ['Mentions Economics', 'Releases Code', 'Tests Real Systems']
    
    x = np.arange(len(years))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for year in years:
            year_df = df[df['Year'] == year]
            pct = (year_df[metric] == 'YES').sum() / len(year_df) * 100
            values.append(pct)
        
        ax.bar(x + i * width, values, width, label=label, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Practical Indicators by Year (Higher is Better)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(years)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add baseline line for reference
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_practical_indicators.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_practical_indicators.pdf', bbox_inches='tight')
    print("Generated: fig4_practical_indicators.png/pdf")
    plt.close()

def plot_gap_indicators(df, stats, output_dir):
    """Figure 5: Research-practice gap indicators (academic characteristics)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Gap indicators by year
    years = sorted(df['Year'].unique())
    gap_metrics = ['Flag_Grad', 'Flag_HighQ', 'Flag_WB', 'Flag_NoCode', 'Flag_NoReal']
    gap_labels = ['Gradient-based', 'High Queries (>1000)', 'White-box', 
                  'No Code Released', 'No Real System Test']
    
    x = np.arange(len(years))
    width = 0.15
    
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (metric, label, color) in enumerate(zip(gap_metrics, gap_labels, colors)):
        values = []
        for year in years:
            year_df = df[df['Year'] == year]
            pct = year_df[metric].sum() / len(year_df) * 100
            values.append(pct)
        
        ax.bar(x + i * width, values, width, label=label, color=color, 
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Research-Practice Gap Indicators by Year (Lower is Better)')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(years)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_gap_indicators.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig5_gap_indicators.pdf', bbox_inches='tight')
    print("Generated: fig5_gap_indicators.png/pdf")
    plt.close()

def plot_comparison_apruzzese(df, stats, output_dir):
    """Figure 6: Comparison with Apruzzese 2019-2021 baseline"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Metrics to compare
    comparisons = [
        ('Attack focus', APRUZZESE_BASELINE['Attack focus'], stats['attack_pct']),
        ('DL only', APRUZZESE_BASELINE['DL only'], stats['dl_only_pct']),
        ('No economics', APRUZZESE_BASELINE['No economics'], stats['no_econ_pct']),
        ('Code released', APRUZZESE_BASELINE['Code released'], stats['code_yes_pct']),
        ('Real systems', APRUZZESE_BASELINE['Real systems'], stats['real_yes_pct']),
        ('Gradient-based', APRUZZESE_BASELINE['Gradient-based'], stats['grad_pct']),
        ('White-box', APRUZZESE_BASELINE['White-box'], stats['white_box_pct']),
    ]
    
    metrics = [c[0] for c in comparisons]
    baseline_values = [c[1] for c in comparisons]
    current_values = [c[2] for c in comparisons]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='2019-2021 (Apruzzese)',
                   color='lightgray', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, current_values, width, label='2022-2025 (This Study)',
                   color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Add change arrows
    for i, (metric, old, new) in enumerate(comparisons):
        change = new - old
        y_pos = max(old, new) + 5
        if abs(change) > 2:  # Only show significant changes
            arrow_props = dict(arrowstyle='->', lw=1.5, 
                             color='green' if change > 0 else 'red')
            ax.annotate('', xy=(i + width/2, new), xytext=(i - width/2, old),
                       arrowprops=arrow_props, zorder=5)
            # Add change text
            change_text = f'{change:+.1f}%'
            ax.text(i, y_pos, change_text, ha='center', fontsize=7,
                   color='green' if change > 0 else 'red', weight='bold')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Comparison with Apruzzese et al. (2019-2021) Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_comparison_baseline.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig6_comparison_baseline.pdf', bbox_inches='tight')
    print("Generated: fig6_comparison_baseline.png/pdf")
    plt.close()

def plot_gap_score_distribution(df, stats, output_dir):
    """Figure 7: Gap score distribution and evolution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Overall distribution
    gap_scores = df['Traditional_Score'].value_counts().sort_index()
    colors = ['green' if x < 2 else 'orange' if x < 4 else 'red' for x in gap_scores.index]
    
    axes[0].bar(gap_scores.index, gap_scores.values, color=colors, 
                edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[0].set_xlabel('Traditional Score (Gap Score)')
    axes[0].set_ylabel('Number of Papers')
    axes[0].set_title('(a) Gap Score Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticks(range(7))
    
    # Add value labels
    for i, v in zip(gap_scores.index, gap_scores.values):
        axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Low Gap (0-1)'),
        Patch(facecolor='orange', edgecolor='black', label='Medium Gap (2-3)'),
        Patch(facecolor='red', edgecolor='black', label='High Gap (4-6)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    # Right: Gap categories by year
    years = sorted(df['Year'].unique())
    high_gap = []
    med_gap = []
    low_gap = []
    
    for year in years:
        year_df = df[df['Year'] == year]
        high_gap.append((year_df['Traditional_Score'] >= 4).sum())
        med_gap.append(((year_df['Traditional_Score'] >= 2) & 
                       (year_df['Traditional_Score'] < 4)).sum())
        low_gap.append((year_df['Traditional_Score'] < 2).sum())
    
    x = np.arange(len(years))
    width = 0.6
    
    axes[1].bar(x, low_gap, width, label='Low Gap (0-1)', color='green', 
                edgecolor='black', linewidth=0.5)
    axes[1].bar(x, med_gap, width, bottom=low_gap, label='Medium Gap (2-3)', 
                color='orange', edgecolor='black', linewidth=0.5)
    axes[1].bar(x, np.array(high_gap), width, 
                bottom=np.array(low_gap) + np.array(med_gap),
                label='High Gap (4-6)', color='red', edgecolor='black', linewidth=0.5)
    
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Number of Papers')
    axes[1].set_title('(b) Gap Categories by Year')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_gap_score_distribution.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig7_gap_score_distribution.pdf', bbox_inches='tight')
    print("Generated: fig7_gap_score_distribution.png/pdf")
    plt.close()

def plot_threat_model(df, output_dir):
    """Figure 8: Threat model assumptions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: T1 - Model knowledge
    t1_counts = df['T1'].value_counts()
    # Remove NA if present
    t1_counts = t1_counts[t1_counts.index != 'NA']
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red for white-box, orange for gray, green for black
    axes[0].bar(t1_counts.index, t1_counts.values, color=colors, 
                edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[0].set_xlabel('Model Knowledge')
    axes[0].set_ylabel('Number of Papers')
    axes[0].set_title('(a) Attacker Model Knowledge (T1)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentages
    total = t1_counts.sum()
    for i, (k, v) in enumerate(t1_counts.items()):
        pct = v / total * 100
        axes[0].text(i, v + 1, f'{v}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # Right: Training data access by year
    years = sorted(df['Year'].unique())
    t2_by_year = df.groupby(['Year', 'T2']).size().unstack(fill_value=0)
    # Remove NA if present
    if 'NA' in t2_by_year.columns:
        t2_by_year = t2_by_year.drop('NA', axis=1)
    
    t2_by_year_pct = t2_by_year.div(t2_by_year.sum(axis=1), axis=0) * 100
    
    t2_by_year_pct.plot(kind='bar', stacked=True, ax=axes[1],
                        color=['#2ca02c', '#ff7f0e', '#d62728'],
                        edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('(b) Training Data Access (T2) by Year')
    axes[1].legend(title='Access Level')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig8_threat_model.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig8_threat_model.pdf', bbox_inches='tight')
    print("Generated: fig8_threat_model.png/pdf")
    plt.close()

def plot_correlation_heatmap(df, output_dir):
    """Figure 9: Correlation between gap indicators"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Select relevant columns
    gap_cols = ['Flag_Grad', 'Flag_HighQ', 'Flag_WB', 'Flag_NoEcon', 
                'Flag_NoCode', 'Flag_NoReal', 'Traditional_Score']
    
    # Calculate correlation
    corr = df[gap_cols].corr()
    
    # Create heatmap
    labels = ['Gradient\nBased', 'High\nQueries', 'White-box', 'No\nEconomics', 
              'No\nCode', 'No Real\nSystem', 'Gap\nScore']
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title('Correlation Between Gap Indicators')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig9_correlation_heatmap.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig9_correlation_heatmap.pdf', bbox_inches='tight')
    print("Generated: fig9_correlation_heatmap.png/pdf")
    plt.close()

def plot_yearly_gap_evolution(df, output_dir):
    """Figure 10: Average gap score evolution over years"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    years = sorted(df['Year'].unique())
    avg_scores = []
    std_scores = []
    
    for year in years:
        year_df = df[df['Year'] == year]
        avg_scores.append(year_df['Traditional_Score'].mean())
        std_scores.append(year_df['Traditional_Score'].std())
    
    # Plot line with error bars
    ax.errorbar(years, avg_scores, yerr=std_scores, marker='o', markersize=8,
                linewidth=2, capsize=5, capthick=2, color='steelblue',
                ecolor='lightblue', label='Average Gap Score')
    
    # Add trend line
    z = np.polyfit(years, avg_scores, 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), linestyle='--', color='red', linewidth=2,
            label=f'Trend (slope={z[0]:.3f})')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Gap Score')
    ax.set_title('Evolution of Research-Practice Gap (2022-2025)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 6)
    
    # Add horizontal reference lines
    ax.axhline(y=2, color='green', linestyle=':', alpha=0.5, label='Low Gap threshold')
    ax.axhline(y=4, color='red', linestyle=':', alpha=0.5, label='High Gap threshold')
    
    # Add value labels
    for year, score in zip(years, avg_scores):
        ax.text(year, score + 0.15, f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig10_gap_evolution.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig10_gap_evolution.pdf', bbox_inches='tight')
    print("Generated: fig10_gap_evolution.png/pdf")
    plt.close()

def generate_summary_table(df, stats, output_dir):
    """Generate summary statistics table"""
    summary_data = {
        'Metric': [
            'Total Papers',
            'Attack Papers',
            'Defense Papers',
            'Deep Learning Only',
            'Mentions Economics',
            'Releases Code',
            'Tests Real Systems',
            'Gradient-based',
            'High Query Budget',
            'White-box Assumption',
            'Average Gap Score',
            'High Gap Papers (4-6)',
            'Low Gap Papers (0-1)',
        ],
        'Value': [
            f"{stats['total_papers']}",
            f"{stats['attack_pct']:.1f}%",
            f"{stats['defense_pct']:.1f}%",
            f"{stats['dl_only_pct']:.1f}%",
            f"{stats['econ_yes_pct']:.1f}%",
            f"{stats['code_yes_pct']:.1f}%",
            f"{stats['real_yes_pct']:.1f}%",
            f"{stats['grad_pct']:.1f}%",
            f"{stats['high_query_pct']:.1f}%",
            f"{stats['white_box_pct']:.1f}%",
            f"{stats['avg_gap_score']:.2f}/6",
            f"{stats['high_gap']} ({stats['high_gap']/stats['total_papers']*100:.1f}%)",
            f"{stats['low_gap']} ({stats['low_gap']/stats['total_papers']*100:.1f}%)",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
    print("\nGenerated: summary_statistics.csv")
    
    # Print to console
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

def main():
    """Main execution function"""
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    
    # Configuration - use relative paths
    CSV_PATH = script_dir / 'analysis_results_full_papers.csv'
    # Fallback to other CSV files if the primary one doesn't exist
    if not CSV_PATH.exists():
        CSV_PATH = script_dir / 'analysis_results.csv'
    
    OUTPUT_DIR = script_dir / 'outputs' / 'figures'
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ADVERSARIAL ML RESEARCH-PRACTICE GAP ANALYSIS")
    print("Generating Publication-Quality Visualizations")
    print("="*60)
    
    # Load data
    df = load_data(CSV_PATH)
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Generate all figures
    print("\nGenerating figures...")
    plot_yearly_trends(df, OUTPUT_DIR)
    plot_attack_types(df, OUTPUT_DIR)
    plot_data_types(df, OUTPUT_DIR)
    plot_practical_indicators(df, stats, OUTPUT_DIR)
    plot_gap_indicators(df, stats, OUTPUT_DIR)
    plot_comparison_apruzzese(df, stats, OUTPUT_DIR)
    plot_gap_score_distribution(df, stats, OUTPUT_DIR)
    plot_threat_model(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_yearly_gap_evolution(df, OUTPUT_DIR)
    
    # Generate summary table
    generate_summary_table(df, stats, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - 10 figures (PNG and PDF formats)")
    print("  - 1 summary statistics CSV")
    print("\nTotal: 21 files")

if __name__ == "__main__":
    main()