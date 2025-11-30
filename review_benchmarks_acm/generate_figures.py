"""
Generate publication-quality figures for Adversarial ML Research Paper
Based on ACM CCS 2022-2025 analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get script directory and set up paths
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "analysis_results_clean.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['patch.linewidth'] = 1

# Color palette - professional and distinguishable
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent1': '#F18F01',      # Orange
    'accent2': '#C73E1D',      # Red
    'neutral1': '#6A994E',     # Green
    'neutral2': '#BC4B51',     # Dark red
    'gray1': '#5D5D5D',        # Dark gray
    'gray2': '#8B8B8B',        # Medium gray
    'attack': '#C73E1D',       # Red for attacks
    'defense': '#2E86AB',      # Blue for defenses
    'both': '#6A994E',         # Green for both
}

def load_data(filepath):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert year to integer
    df['Year'] = df['Year'].astype(int)
    
    # Create binary flags for easier analysis
    df['Is_Attack'] = df['G1'].str.lower() == 'atk'
    df['Is_Defense'] = df['G1'].str.lower() == 'def'
    df['Is_Both'] = df['G1'].str.lower() == 'both'
    
    return df

def figure1_temporal_trends(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 1: Temporal Trends in Key Metrics (2022-2025)
    Shows evolution of gradient-based, high query, white-box, and real-world testing
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate metrics by year
    metrics_by_year = df.groupby('Year').agg({
        'Flag_Grad': lambda x: (x == 1).sum() / len(x) * 100,
        'Flag_HighQ': lambda x: (x == 1).sum() / len(x) * 100,
        'Flag_WB': lambda x: (x == 1).sum() / len(x) * 100,
        'Flag_NoReal': lambda x: ((x == 0).sum() / len(x)) * 100,  # Inverted: papers WITH real testing
    }).reset_index()
    
    years = metrics_by_year['Year']
    x = np.arange(len(years))
    
    # Plot lines
    ax.plot(x, metrics_by_year['Flag_Grad'], marker='o', label='Gradient-Based', 
            color=COLORS['accent2'], linewidth=2.5, markersize=8)
    ax.plot(x, metrics_by_year['Flag_HighQ'], marker='s', label='High Query Budget (>1000)', 
            color=COLORS['accent1'], linewidth=2.5, markersize=8)
    ax.plot(x, metrics_by_year['Flag_WB'], marker='^', label='White-Box Access', 
            color=COLORS['secondary'], linewidth=2.5, markersize=8)
    ax.plot(x, metrics_by_year['Flag_NoReal'], marker='d', label='Real-World Testing', 
            color=COLORS['neutral1'], linewidth=2.5, markersize=8)
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Temporal Trends in Research Characteristics (2022-2025)', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure1_temporal_trends.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure1_temporal_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 1: Temporal Trends")
    plt.close()

def figure2_threat_model_distribution(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 2: Threat Model Distribution Across Years
    Stacked bar chart showing white-box, gray-box, black-box, and multiple threat models
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Count threat models by year
    threat_counts = df.groupby(['Year', 'T1']).size().unstack(fill_value=0)
    
    # Calculate percentages
    threat_percentages = threat_counts.div(threat_counts.sum(axis=1), axis=0) * 100
    
    # Reorder columns if they exist
    desired_order = ['White-box', 'Gray-box', 'Black-box', 'Multiple']
    existing_cols = [col for col in desired_order if col in threat_percentages.columns]
    threat_percentages = threat_percentages[existing_cols]
    
    # Define colors for each threat model
    colors_threat = {
        'White-box': COLORS['accent2'],
        'Gray-box': COLORS['accent1'],
        'Black-box': COLORS['primary'],
        'Multiple': COLORS['neutral1']
    }
    
    # Create stacked bar chart
    threat_percentages.plot(kind='bar', stacked=True, ax=ax, 
                           color=[colors_threat.get(col, COLORS['gray1']) for col in threat_percentages.columns],
                           width=0.7, edgecolor='black', linewidth=1.2)
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Distribution of Threat Models (2022-2025)', fontweight='bold', pad=15)
    ax.set_xticklabels(threat_percentages.index, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend(title='Threat Model', title_fontsize=11, framealpha=0.95, 
             edgecolor='black', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure2_threat_models.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure2_threat_models.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 2: Threat Model Distribution")
    plt.close()

def figure3_attack_defense_distribution(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 3: Attack vs Defense Papers Distribution by Year
    Shows the imbalance between offensive and defensive research
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count by year and type
    type_counts = df.groupby(['Year', 'G1']).size().unstack(fill_value=0)
    
    # Ensure all categories exist
    for cat in ['atk', 'def', 'both']:
        if cat not in type_counts.columns:
            type_counts[cat] = 0
    
    type_counts = type_counts[['atk', 'def', 'both']]
    type_counts.columns = ['Attack', 'Defense', 'Both']
    
    # Plot 1: Stacked bar chart
    type_counts.plot(kind='bar', stacked=False, ax=ax1, 
                    color=[COLORS['attack'], COLORS['defense'], COLORS['both']],
                    width=0.7, edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontweight='bold')
    ax1.set_title('Attack vs Defense Papers by Year', fontweight='bold', pad=15)
    ax1.set_xticklabels(type_counts.index, rotation=0)
    ax1.legend(title='Paper Type', framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot 2: Overall distribution (pie chart)
    overall_counts = df['G1'].value_counts()
    overall_counts.index = overall_counts.index.str.capitalize()
    
    colors_pie = [COLORS['attack'] if 'Atk' in idx else 
                  COLORS['defense'] if 'Def' in idx else 
                  COLORS['both'] for idx in overall_counts.index]
    
    wedges, texts, autotexts = ax2.pie(overall_counts, labels=overall_counts.index, 
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors_pie, explode=[0.05]*len(overall_counts),
                                        textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    ax2.set_title('Overall Distribution (2022-2025)', fontweight='bold', pad=15)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure3_attack_defense.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure3_attack_defense.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 3: Attack vs Defense Distribution")
    plt.close()

def figure4_key_metrics_comparison(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 4: Comparison of Key Research Characteristics
    Grouped bar chart comparing multiple metrics
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Calculate overall percentages
    metrics = {
        'Gradient-Based': (df['Flag_Grad'] == 1).sum() / len(df) * 100,
        'High Query\nBudget (>1000)': (df['Flag_HighQ'] == 1).sum() / len(df) * 100,
        'White-Box\nAccess': (df['Flag_WB'] == 1).sum() / len(df) * 100,
        'No Economic\nConsiderations': (df['Flag_NoEcon'] == 1).sum() / len(df) * 100,
        'No Code\nReleased': (df['Flag_NoCode'] == 1).sum() / len(df) * 100,
        'No Real-World\nTesting': (df['Flag_NoReal'] == 1).sum() / len(df) * 100,
    }
    
    # Create bar chart
    x_pos = np.arange(len(metrics))
    values = list(metrics.values())
    labels = list(metrics.keys())
    
    # Color bars based on whether they're positive or negative indicators
    colors_bars = [COLORS['accent2'] if val > 50 else COLORS['primary'] for val in values]
    
    bars = ax.bar(x_pos, values, color=colors_bars, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')
    
    # Formatting
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Key Research Characteristics (2022-2025, N=105)', fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure4_key_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure4_key_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 4: Key Metrics Comparison")
    plt.close()

def figure5_attack_type_distribution(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 5: Distribution of Attack/Defense Types
    Shows Privacy, Evasion, Poisoning, etc.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Count attack types
    attack_types = df['G2'].value_counts().head(10)  # Top 10
    
    # Create color palette
    colors_attacks = plt.cm.Set3(np.linspace(0, 1, len(attack_types)))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(attack_types))
    bars = ax.barh(y_pos, attack_types.values, color=colors_attacks, 
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, attack_types.values)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2., 
                f'{int(val)}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Number of Papers', fontweight='bold')
    ax.set_title('Distribution of Attack/Defense Types (2022-2025)', fontweight='bold', pad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attack_types.index)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure5_attack_types.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure5_attack_types.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 5: Attack Type Distribution")
    plt.close()

def figure6_domain_distribution(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 6: Application Domain Distribution
    Shows Images, Text, Audio, Malware, etc.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count domains
    domain_counts = df['G4'].value_counts()
    
    # Plot 1: Bar chart
    colors_domain = plt.cm.Paired(np.linspace(0, 1, len(domain_counts)))
    
    x_pos = np.arange(len(domain_counts))
    bars = ax1.bar(x_pos, domain_counts.values, color=colors_domain, 
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Application Domain', fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontweight='bold')
    ax1.set_title('Application Domain Distribution', fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(domain_counts.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot 2: Evolution over time (stacked area for top domains)
    top_domains = domain_counts.head(5).index
    domain_by_year = df[df['G4'].isin(top_domains)].groupby(['Year', 'G4']).size().unstack(fill_value=0)
    
    ax2.stackplot(domain_by_year.index, 
                 *[domain_by_year[col] for col in domain_by_year.columns],
                 labels=domain_by_year.columns,
                 colors=colors_domain[:len(domain_by_year.columns)],
                 alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Number of Papers', fontweight='bold')
    ax2.set_title('Domain Evolution Over Time (Top 5)', fontweight='bold', pad=15)
    ax2.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure6_domains.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure6_domains.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 6: Domain Distribution")
    plt.close()

def figure7_dl_vs_traditional(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 7: Deep Learning vs Traditional ML Focus
    Shows the overwhelming focus on DL systems
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Count by year and ML type
    ml_type_by_year = df.groupby(['Year', 'G3']).size().unstack(fill_value=0)
    
    # Calculate percentages
    ml_type_pct = ml_type_by_year.div(ml_type_by_year.sum(axis=1), axis=0) * 100
    
    # Create stacked bar chart
    x = np.arange(len(ml_type_pct.index))
    width = 0.6
    
    colors_ml = {
        'DL': COLORS['primary'],
        'Traditional': COLORS['accent1'],
        'Both': COLORS['neutral1']
    }
    
    bottom = np.zeros(len(ml_type_pct.index))
    for col in ml_type_pct.columns:
        if col in colors_ml:
            col_values = ml_type_pct[col].values  # Convert to numpy array
            ax.bar(x, col_values, width, label=col, bottom=bottom,
                  color=colors_ml[col], edgecolor='black', linewidth=1.2)
            
            # Add percentage labels
            for i, val in enumerate(col_values):
                if val > 5:  # Only show if > 5%
                    ax.text(x[i], bottom[i] + val/2, f'{val:.1f}%',
                           ha='center', va='center', fontweight='bold', 
                           color='white', fontsize=10)
            
            bottom += col_values
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Deep Learning vs Traditional ML Focus (2022-2025)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ml_type_pct.index)
    ax.set_ylim(0, 100)
    ax.legend(title='ML Type', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure7_dl_focus.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure7_dl_focus.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 7: DL vs Traditional ML")
    plt.close()

def figure8_code_release_trend(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Figure 8: Code Release Trends Over Time
    Shows improvement in reproducibility
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate code release percentage by year
    code_by_year = df.groupby('Year').agg({
        'Flag_NoCode': lambda x: ((x == 0).sum() / len(x)) * 100,  # Inverted: papers WITH code
    }).reset_index()
    code_by_year.columns = ['Year', 'Code_Released_Pct']
    
    # Create bar chart
    x = np.arange(len(code_by_year))
    bars = ax.bar(x, code_by_year['Code_Released_Pct'], 
                  color=COLORS['neutral1'], edgecolor='black', linewidth=1.5,
                  alpha=0.8, width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Add trend line
    z = np.polyfit(code_by_year.index, code_by_year['Code_Released_Pct'], 1)
    p = np.poly1d(z)
    ax.plot(x, p(code_by_year.index), "r--", linewidth=2, 
            label=f'Trend: {z[0]:.1f}% increase per year', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Papers with Code Released (%)', fontweight='bold')
    ax.set_title('Code Release Trends (2022-2025)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(code_by_year['Year'])
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}figure8_code_release.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}figure8_code_release.png', dpi=300, bbox_inches='tight')
    print("✓ Generated Figure 8: Code Release Trends")
    plt.close()

def generate_summary_stats(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Generate a summary statistics table
    """
    stats = {
        'Total Papers': len(df),
        'Years Covered': f"{df['Year'].min()}-{df['Year'].max()}",
        'Attack Papers': (df['G1'] == 'atk').sum(),
        'Defense Papers': (df['G1'] == 'def').sum(),
        'Both Papers': (df['G1'] == 'both').sum(),
        'Gradient-Based (%)': f"{(df['Flag_Grad'] == 1).sum() / len(df) * 100:.1f}",
        'White-Box (%)': f"{(df['Flag_WB'] == 1).sum() / len(df) * 100:.1f}",
        'Black-Box (%)': f"{(df['T1'] == 'Black-box').sum() / len(df) * 100:.1f}",
        'High Query Budget (%)': f"{(df['Flag_HighQ'] == 1).sum() / len(df) * 100:.1f}",
        'Real-World Testing (%)': f"{(df['Flag_NoReal'] == 0).sum() / len(df) * 100:.1f}",
        'Code Released (%)': f"{(df['Flag_NoCode'] == 0).sum() / len(df) * 100:.1f}",
        'Economic Considerations (%)': f"{(df['Flag_NoEcon'] == 0).sum() / len(df) * 100:.1f}",
    }
    
    # Save to text file
    with open(f'{output_path}summary_statistics.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("SUMMARY STATISTICS - ACM CCS 2022-2025 Analysis\n")
        f.write("="*60 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key:.<40} {value:>15}\n")
        f.write("\n" + "="*60 + "\n")
    
    print("✓ Generated Summary Statistics")
    return stats

def main():
    """Main function to generate all figures"""
    print("\n" + "="*60)
    print("Generating Publication-Quality Figures")
    print("ACM CCS 2022-2025 Adversarial ML Analysis")
    print("="*60 + "\n")
    
    # Load data
    df = load_data(str(CSV_FILE))
    print(f"Loaded {len(df)} papers from {df['Year'].min()} to {df['Year'].max()}\n")
    
    # Generate all figures
    figure1_temporal_trends(df)
    figure2_threat_model_distribution(df)
    figure3_attack_defense_distribution(df)
    figure4_key_metrics_comparison(df)
    figure5_attack_type_distribution(df)
    figure6_domain_distribution(df)
    figure7_dl_vs_traditional(df)
    figure8_code_release_trend(df)
    
    # Generate summary statistics
    stats = generate_summary_stats(df)
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  • Figure 1: Temporal Trends (figure1_temporal_trends.pdf/.png)")
    print("  • Figure 2: Threat Models (figure2_threat_models.pdf/.png)")
    print("  • Figure 3: Attack vs Defense (figure3_attack_defense.pdf/.png)")
    print("  • Figure 4: Key Metrics (figure4_key_metrics.pdf/.png)")
    print("  • Figure 5: Attack Types (figure5_attack_types.pdf/.png)")
    print("  • Figure 6: Domain Distribution (figure6_domains.pdf/.png)")
    print("  • Figure 7: DL vs Traditional (figure7_dl_focus.pdf/.png)")
    print("  • Figure 8: Code Release (figure8_code_release.pdf/.png)")
    print("  • Summary Statistics (summary_statistics.txt)")
    print("\n")

if __name__ == "__main__":
    main()