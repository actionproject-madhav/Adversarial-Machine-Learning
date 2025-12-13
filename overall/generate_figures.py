"""
Adversarial ML Research-Practice Gap Analysis
Generates 10 publication-ready figures from conference paper analysis

Author: [Your Name]
Data: 454 papers from ACM CCS, IEEE S&P, NDSS, USENIX Security (2022-2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - UPDATE THIS TO YOUR CSV PATH
INPUT_CSV = "all_conferences_analysis_results_2022_2025.csv"
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

# Color palettes
CONFERENCE_COLORS = {
    'ACM': '#2E86AB',      # Blue
    'IEEE': '#E94F37',     # Red  
    'NDSS': '#F39C12',     # Orange (changed from NDS to NDSS for clarity)
    'NDS': '#F39C12',      # Also map NDS 
    'USENIX': '#27AE60'    # Green
}

# For sequential data (gaps/problems) - red spectrum
GAP_COLORS = ['#FADBD8', '#F1948A', '#E74C3C', '#B03A2E', '#78281F', '#4A1A12']

# For practical indicators (good things) - green/blue spectrum  
PRACTICAL_COLORS = ['#2E86AB', '#27AE60', '#8E44AD']

# For threat models
THREAT_MODEL_COLORS = {
    'White-box': '#E74C3C',   # Red (unrealistic)
    'Gray-box': '#F39C12',    # Orange (middle)
    'Black-box': '#27AE60'    # Green (realistic)
}

# For attack types
ATTACK_TYPE_COLORS = {
    'Evasion': '#3498DB',
    'Poisoning': '#E74C3C', 
    'Privacy': '#9B59B6',
    'Multiple': '#F39C12',
    'NA': '#95A5A6'
}

# For data domains
DOMAIN_COLORS = {
    'Images': '#3498DB',
    'Text': '#E74C3C',
    'Audio': '#9B59B6',
    'Malware': '#2C3E50',
    'Other': '#95A5A6'
}

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(filepath):
    """Load CSV and prepare data for analysis."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Standardize conference names (handle NDS vs NDSS)
    df['Conference'] = df['Conference'].replace({'NDS': 'NDSS'})
    
    # Ensure Year is integer
    df['Year'] = df['Year'].astype(int)
    
    # Fill empty values
    df = df.fillna('')
    
    # Print basic info
    print(f"\nDataset Summary:")
    print(f"  Total papers: {len(df)}")
    print(f"  Conferences: {df['Conference'].value_counts().to_dict()}")
    print(f"  Years: {sorted(df['Year'].unique())}")
    
    return df

# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_all_statistics(df):
    """Calculate and print all key statistics."""
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICS")
    print("="*70)
    
    total = len(df)
    
    # Overall gap prevalence
    print("\n--- OVERALL GAP INDICATORS (% of all papers) ---")
    gap_stats = {
        'Ignores Economics (G5=NO)': (df['G5'] == 'NO').sum() / total * 100,
        'No Code Released (G6=NO)': (df['G6'] == 'NO').sum() / total * 100,
        'No Real System Testing (G7=NO)': (df['G7'] == 'NO').sum() / total * 100,
        'Requires Gradients (Q1=YES)': (df['Q1'] == 'YES').sum() / total * 100,
        'High Query Budget (Q2=High)': (df['Q2'] == 'High').sum() / total * 100,
        'White-box Only (T1=White-box)': (df['T1'] == 'White-box').sum() / total * 100,
    }
    for name, pct in gap_stats.items():
        print(f"  {name}: {pct:.1f}%")
    
    # Practical indicators (inverse)
    print("\n--- PRACTICAL INDICATORS (% of all papers) ---")
    practical_stats = {
        'Considers Economics (G5=YES)': (df['G5'] == 'YES').sum() / total * 100,
        'Releases Code (G6=YES)': (df['G6'] == 'YES').sum() / total * 100,
        'Tests Real Systems (G7=YES)': (df['G7'] == 'YES').sum() / total * 100,
    }
    for name, pct in practical_stats.items():
        print(f"  {name}: {pct:.1f}%")
    
    # Traditional Score statistics
    print("\n--- TRADITIONAL SCORE (Gap Score, 0-6) ---")
    print(f"  Mean: {df['Traditional_Score'].mean():.2f}")
    print(f"  Median: {df['Traditional_Score'].median():.1f}")
    print(f"  Std Dev: {df['Traditional_Score'].std():.2f}")
    print(f"  Min: {df['Traditional_Score'].min()}, Max: {df['Traditional_Score'].max()}")
    
    # By conference
    print("\n--- TRADITIONAL SCORE BY CONFERENCE ---")
    for conf in ['ACM', 'IEEE', 'NDSS', 'USENIX']:
        conf_df = df[df['Conference'] == conf]
        if len(conf_df) > 0:
            print(f"  {conf}: Mean={conf_df['Traditional_Score'].mean():.2f}, "
                  f"Median={conf_df['Traditional_Score'].median():.1f}, "
                  f"n={len(conf_df)}")
    
    # By year
    print("\n--- TRADITIONAL SCORE BY YEAR ---")
    for year in sorted(df['Year'].unique()):
        year_df = df[df['Year'] == year]
        print(f"  {year}: Mean={year_df['Traditional_Score'].mean():.2f}, n={len(year_df)}")
    
    # Threat model distribution
    print("\n--- THREAT MODEL DISTRIBUTION (T1) ---")
    t1_counts = df['T1'].value_counts()
    for model, count in t1_counts.items():
        print(f"  {model}: {count} ({count/total*100:.1f}%)")
    
    # Attack type distribution
    print("\n--- ATTACK TYPE DISTRIBUTION (G2) ---")
    g2_counts = df['G2'].value_counts()
    for attack, count in g2_counts.items():
        print(f"  {attack}: {count} ({count/total*100:.1f}%)")
    
    # Focus distribution
    print("\n--- RESEARCH FOCUS (G1) ---")
    g1_counts = df['G1'].value_counts()
    for focus, count in g1_counts.items():
        print(f"  {focus}: {count} ({count/total*100:.1f}%)")
    
    return gap_stats, practical_stats

# ============================================================================
# FIGURE 1: Dataset Overview - Papers by Conference × Year
# ============================================================================

def figure_1_dataset_overview(df):
    """Create grouped bar chart showing papers by conference and year."""
    print("\nGenerating Figure 1: Dataset Overview...")
    
    # Prepare data
    conf_year = df.groupby(['Year', 'Conference']).size().unstack(fill_value=0)
    
    # Ensure consistent conference order
    conf_order = ['ACM', 'IEEE', 'NDSS', 'USENIX']
    conf_year = conf_year.reindex(columns=[c for c in conf_order if c in conf_year.columns])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar positions
    x = np.arange(len(conf_year.index))
    width = 0.2
    multiplier = 0
    
    # Plot bars for each conference
    for conf in conf_year.columns:
        offset = width * multiplier
        bars = ax.bar(x + offset, conf_year[conf], width, 
                      label=conf, color=CONFERENCE_COLORS.get(conf, '#888888'),
                      edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        multiplier += 1
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Number of Papers', fontweight='bold')
    ax.set_title('Distribution of Adversarial ML Papers by Conference and Year\n(2022-2025)', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(conf_year.index)
    ax.legend(title='Conference', loc='upper left', framealpha=0.9)
    ax.set_ylim(0, conf_year.values.max() * 1.2)
    
    # Add total annotation
    total = len(df)
    ax.text(0.98, 0.98, f'Total: {total} papers', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_dataset_overview.png')
    plt.savefig(OUTPUT_DIR / 'fig1_dataset_overview.pdf')
    plt.close()
    print("  Saved: fig1_dataset_overview.png/pdf")

# ============================================================================
# FIGURE 2: Overall Gap Indicators
# ============================================================================

def figure_2_overall_gap(df):
    """Create horizontal bar chart showing % of papers with each gap indicator."""
    print("\nGenerating Figure 2: Overall Gap Indicators...")
    
    total = len(df)
    
    # Calculate percentages (showing the GAP/PROBLEM side)
    gap_data = {
        'No Real System Testing': (df['G7'] == 'NO').sum() / total * 100,
        'No Code Released': (df['G6'] == 'NO').sum() / total * 100,
        'Ignores Economics': (df['G5'] == 'NO').sum() / total * 100,
        'Requires Gradients': (df['Q1'] == 'YES').sum() / total * 100,
        'White-box Only': (df['T1'] == 'White-box').sum() / total * 100,
        'High Query Budget': (df['Q2'] == 'High').sum() / total * 100,
    }
    
    # Sort by percentage
    gap_data = dict(sorted(gap_data.items(), key=lambda x: x[1], reverse=True))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(gap_data.keys())
    values = list(gap_data.values())
    y_pos = np.arange(len(labels))
    
    # Create gradient colors based on severity
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(values)))
    # Sort colors to match sorted data (highest gets darkest)
    
    # Plot horizontal bars
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Theory-Practice Gap: Prevalence of Problematic Research Patterns\n(n={} papers)'.format(total),
                 fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    
    # Add vertical line at 50%
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(51, len(labels)-0.5, '50%', color='gray', fontsize=9)
    
    # Add grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_overall_gap_indicators.png')
    plt.savefig(OUTPUT_DIR / 'fig2_overall_gap_indicators.pdf')
    plt.close()
    print("  Saved: fig2_overall_gap_indicators.png/pdf")

# ============================================================================
# FIGURE 3: Gap Indicators by Conference (Heatmap)
# ============================================================================

def figure_3_gap_by_conference_heatmap(df):
    """Create heatmap showing gap indicators across conferences."""
    print("\nGenerating Figure 3: Gap Indicators by Conference (Heatmap)...")
    
    conferences = ['ACM', 'IEEE', 'NDSS', 'USENIX']
    indicators = [
        ('No Real System\nTesting', 'G7', 'NO'),
        ('No Code\nReleased', 'G6', 'NO'),
        ('Ignores\nEconomics', 'G5', 'NO'),
        ('Requires\nGradients', 'Q1', 'YES'),
        ('White-box\nOnly', 'T1', 'White-box'),
        ('High Query\nBudget', 'Q2', 'High'),
    ]
    
    # Build data matrix
    data = []
    for conf in conferences:
        conf_df = df[df['Conference'] == conf]
        total = len(conf_df)
        row = []
        for name, col, val in indicators:
            if total > 0:
                pct = (conf_df[col] == val).sum() / total * 100
            else:
                pct = 0
            row.append(pct)
        data.append(row)
    
    data = np.array(data)
    indicator_names = [ind[0] for ind in indicators]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create heatmap
    im = ax.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom", fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(len(indicator_names)))
    ax.set_yticks(np.arange(len(conferences)))
    ax.set_xticklabels(indicator_names, fontsize=10)
    ax.set_yticklabels(conferences, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), ha="center")
    
    # Add text annotations
    for i in range(len(conferences)):
        for j in range(len(indicator_names)):
            val = data[i, j]
            # Choose text color based on background
            text_color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                   color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_title('Gap Indicators by Conference\n(Higher % = Larger Theory-Practice Gap)',
                 fontweight='bold', pad=15)
    
    # Add conference paper counts
    for i, conf in enumerate(conferences):
        n = len(df[df['Conference'] == conf])
        ax.text(-0.7, i, f'(n={n})', ha='right', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_gap_by_conference_heatmap.png')
    plt.savefig(OUTPUT_DIR / 'fig3_gap_by_conference_heatmap.pdf')
    plt.close()
    print("  Saved: fig3_gap_by_conference_heatmap.png/pdf")

# ============================================================================
# FIGURE 4: Practical Indicators Over Years (Line Chart)
# ============================================================================

def figure_4_practical_trends(df):
    """Create line chart showing practical indicators over time."""
    print("\nGenerating Figure 4: Practical Indicators Over Years...")
    
    years = sorted(df['Year'].unique())
    
    # Calculate percentages per year
    economics_pct = []
    code_pct = []
    real_system_pct = []
    
    for year in years:
        year_df = df[df['Year'] == year]
        total = len(year_df)
        if total > 0:
            economics_pct.append((year_df['G5'] == 'YES').sum() / total * 100)
            code_pct.append((year_df['G6'] == 'YES').sum() / total * 100)
            real_system_pct.append((year_df['G7'] == 'YES').sum() / total * 100)
        else:
            economics_pct.append(0)
            code_pct.append(0)
            real_system_pct.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines with markers
    ax.plot(years, economics_pct, 'o-', color='#2E86AB', linewidth=2.5, 
            markersize=10, label='Considers Economics (G5=YES)', markeredgecolor='white')
    ax.plot(years, code_pct, 's-', color='#27AE60', linewidth=2.5,
            markersize=10, label='Releases Code (G6=YES)', markeredgecolor='white')
    ax.plot(years, real_system_pct, '^-', color='#8E44AD', linewidth=2.5,
            markersize=10, label='Tests Real Systems (G7=YES)', markeredgecolor='white')
    
    # Add value labels
    for i, year in enumerate(years):
        ax.annotate(f'{economics_pct[i]:.1f}%', (year, economics_pct[i]), 
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#2E86AB')
        ax.annotate(f'{code_pct[i]:.1f}%', (year, code_pct[i]),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#27AE60')
        ax.annotate(f'{real_system_pct[i]:.1f}%', (year, real_system_pct[i]),
                   textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='#8E44AD')
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Trend of Practical Research Indicators (2022-2025)\n(Higher = More Practical)',
                 fontweight='bold', pad=15)
    ax.set_xticks(years)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_practical_indicators_trend.png')
    plt.savefig(OUTPUT_DIR / 'fig4_practical_indicators_trend.pdf')
    plt.close()
    print("  Saved: fig4_practical_indicators_trend.png/pdf")

# ============================================================================
# FIGURE 5: Threat Model Distribution by Conference
# ============================================================================

def figure_5_threat_model_distribution(df):
    """Create stacked bar chart showing threat model distribution."""
    print("\nGenerating Figure 5: Threat Model Distribution...")
    
    conferences = ['ACM', 'IEEE', 'NDSS', 'USENIX']
    threat_models = ['White-box', 'Gray-box', 'Black-box']
    
    # Calculate percentages
    data = {tm: [] for tm in threat_models}
    
    for conf in conferences:
        conf_df = df[df['Conference'] == conf]
        total = len(conf_df)
        for tm in threat_models:
            if total > 0:
                pct = (conf_df['T1'] == tm).sum() / total * 100
            else:
                pct = 0
            data[tm].append(pct)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(conferences))
    width = 0.6
    
    # Create stacked bars
    bottom = np.zeros(len(conferences))
    for tm in threat_models:
        bars = ax.bar(x, data[tm], width, label=tm, bottom=bottom,
                     color=THREAT_MODEL_COLORS[tm], edgecolor='white', linewidth=0.5)
        
        # Add percentage labels in the middle of each segment
        for i, (val, bot) in enumerate(zip(data[tm], bottom)):
            if val > 5:  # Only show label if segment is big enough
                ax.text(i, bot + val/2, f'{val:.1f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
        
        bottom += np.array(data[tm])
    
    # Formatting
    ax.set_xlabel('Conference', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Threat Model Assumptions by Conference\n(White-box = Unrealistic Full Model Access)',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={len(df[df["Conference"]==c])})' for c in conferences])
    ax.legend(title='Threat Model (T1)', loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_threat_model_distribution.png')
    plt.savefig(OUTPUT_DIR / 'fig5_threat_model_distribution.pdf')
    plt.close()
    print("  Saved: fig5_threat_model_distribution.png/pdf")

# ============================================================================
# FIGURE 6: Gradient Requirement by Attack Type
# ============================================================================

def figure_6_gradient_by_attack_type(df):
    """Create bar chart showing gradient requirement by attack type."""
    print("\nGenerating Figure 6: Gradient Requirement by Attack Type...")
    
    # Filter to attack types with enough samples
    attack_types = ['Evasion', 'Poisoning', 'Privacy', 'Multiple']
    
    gradient_pct = []
    counts = []
    
    for attack in attack_types:
        attack_df = df[df['G2'] == attack]
        total = len(attack_df)
        counts.append(total)
        if total > 0:
            pct = (attack_df['Q1'] == 'YES').sum() / total * 100
        else:
            pct = 0
        gradient_pct.append(pct)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(attack_types))
    colors = [ATTACK_TYPE_COLORS.get(at, '#888888') for at in attack_types]
    
    bars = ax.bar(x, gradient_pct, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
    
    # Add value labels
    for bar, val, count in zip(bars, gradient_pct, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
               f'n={count}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Attack Type (G2)', fontweight='bold')
    ax.set_ylabel('Papers Requiring Gradients (%)', fontweight='bold')
    ax.set_title('"Real Attackers Don\'t Compute Gradients"\nGradient Dependency by Attack Type',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.set_ylim(0, 100)
    
    # Add reference line
    overall_pct = (df['Q1'] == 'YES').sum() / len(df) * 100
    ax.axhline(y=overall_pct, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(len(attack_types)-0.5, overall_pct + 2, f'Overall: {overall_pct:.1f}%', 
            color='red', fontsize=9, ha='right')
    
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_gradient_by_attack_type.png')
    plt.savefig(OUTPUT_DIR / 'fig6_gradient_by_attack_type.pdf')
    plt.close()
    print("  Saved: fig6_gradient_by_attack_type.png/pdf")

# ============================================================================
# FIGURE 7: Data Domain Distribution
# ============================================================================

def figure_7_domain_distribution(df):
    """Create bar chart showing data domain distribution."""
    print("\nGenerating Figure 7: Data Domain Distribution...")
    
    # Count domains
    domain_counts = df['G4'].value_counts()
    
    # Ensure order
    domains = ['Images', 'Text', 'Audio', 'Malware', 'Other']
    counts = [domain_counts.get(d, 0) for d in domains]
    percentages = [c / len(df) * 100 for c in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(domains))
    colors = [DOMAIN_COLORS.get(d, '#888888') for d in domains]
    
    bars = ax.bar(x, percentages, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
    
    # Add value labels
    for bar, pct, count in zip(bars, percentages, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{pct:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Data Domain (G4)', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Distribution of Research by Data Domain\n(Showing Image-Centric Bias)',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, max(percentages) * 1.25)
    
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_domain_distribution.png')
    plt.savefig(OUTPUT_DIR / 'fig7_domain_distribution.pdf')
    plt.close()
    print("  Saved: fig7_domain_distribution.png/pdf")

# ============================================================================
# FIGURE 8: Traditional Score by Conference (Box Plot)
# ============================================================================

def figure_8_gap_score_by_conference(df):
    """Create box plot showing Traditional_Score distribution by conference."""
    print("\nGenerating Figure 8: Gap Score by Conference...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conferences = ['ACM', 'IEEE', 'NDSS', 'USENIX']
    
    # Prepare data for box plot
    data_to_plot = [df[df['Conference'] == conf]['Traditional_Score'].values for conf in conferences]
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=conferences, patch_artist=True, widths=0.5)
    
    # Color the boxes
    for patch, conf in zip(bp['boxes'], conferences):
        patch.set_facecolor(CONFERENCE_COLORS.get(conf, '#888888'))
        patch.set_alpha(0.7)
    
    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    # Add mean markers
    means = [df[df['Conference'] == conf]['Traditional_Score'].mean() for conf in conferences]
    ax.scatter(range(1, len(conferences)+1), means, color='red', marker='D', s=50, zorder=5, label='Mean')
    
    # Add annotations for mean values
    for i, (conf, mean) in enumerate(zip(conferences, means)):
        n = len(df[df['Conference'] == conf])
        ax.text(i+1, mean + 0.15, f'μ={mean:.2f}', ha='center', va='bottom', fontsize=9, color='red')
        ax.text(i+1, -0.4, f'n={n}', ha='center', va='top', fontsize=9, color='gray')
    
    # Formatting
    ax.set_xlabel('Conference', fontweight='bold')
    ax.set_ylabel('Traditional Score (Gap Score, 0-6)', fontweight='bold')
    ax.set_title('Distribution of Theory-Practice Gap Score by Conference\n(Higher = Larger Gap)',
                 fontweight='bold', pad=15)
    ax.set_ylim(-0.7, 6.5)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_gap_score_by_conference.png')
    plt.savefig(OUTPUT_DIR / 'fig8_gap_score_by_conference.pdf')
    plt.close()
    print("  Saved: fig8_gap_score_by_conference.png/pdf")

# ============================================================================
# FIGURE 9: Traditional Score Over Time
# ============================================================================

def figure_9_gap_score_trend(df):
    """Create line chart showing gap score trend over time."""
    print("\nGenerating Figure 9: Gap Score Trend Over Time...")
    
    years = sorted(df['Year'].unique())
    conferences = ['ACM', 'IEEE', 'NDSS', 'USENIX']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot overall trend
    overall_means = [df[df['Year'] == year]['Traditional_Score'].mean() for year in years]
    ax.plot(years, overall_means, 'ko-', linewidth=3, markersize=12, 
            label='Overall', markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    # Plot per-conference trends
    for conf in conferences:
        conf_means = []
        for year in years:
            year_conf_df = df[(df['Year'] == year) & (df['Conference'] == conf)]
            if len(year_conf_df) > 0:
                conf_means.append(year_conf_df['Traditional_Score'].mean())
            else:
                conf_means.append(np.nan)
        
        ax.plot(years, conf_means, 'o--', color=CONFERENCE_COLORS[conf], 
                linewidth=1.5, markersize=8, label=conf, alpha=0.7)
    
    # Add value labels for overall
    for year, mean in zip(years, overall_means):
        ax.annotate(f'{mean:.2f}', (year, mean), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Mean Traditional Score (0-6)', fontweight='bold')
    ax.set_title('Trend of Theory-Practice Gap Over Time\n(Lower = Narrowing Gap)',
                 fontweight='bold', pad=15)
    ax.set_xticks(years)
    ax.set_ylim(0, 6)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_gap_score_trend.png')
    plt.savefig(OUTPUT_DIR / 'fig9_gap_score_trend.pdf')
    plt.close()
    print("  Saved: fig9_gap_score_trend.png/pdf")

# ============================================================================
# FIGURE 10: Attack vs Defense Practicality
# ============================================================================

def figure_10_attack_vs_defense(df):
    """Create grouped bar comparing practicality of attack vs defense papers."""
    print("\nGenerating Figure 10: Attack vs Defense Practicality...")
    
    # Filter to attack and defense papers
    attack_df = df[df['G1'] == 'atk']
    defense_df = df[df['G1'] == 'def']
    
    indicators = ['Considers\nEconomics', 'Releases\nCode', 'Tests Real\nSystems']
    columns = ['G5', 'G6', 'G7']
    
    attack_pcts = []
    defense_pcts = []
    
    for col in columns:
        attack_pcts.append((attack_df[col] == 'YES').sum() / len(attack_df) * 100 if len(attack_df) > 0 else 0)
        defense_pcts.append((defense_df[col] == 'YES').sum() / len(defense_df) * 100 if len(defense_df) > 0 else 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(indicators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, attack_pcts, width, label=f'Attack Papers (n={len(attack_df)})',
                   color='#E74C3C', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, defense_pcts, width, label=f'Defense Papers (n={len(defense_df)})',
                   color='#3498DB', edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars1, attack_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E74C3C')
    for bar, val in zip(bars2, defense_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#3498DB')
    
    # Formatting
    ax.set_xlabel('Practical Indicator', fontweight='bold')
    ax.set_ylabel('Percentage of Papers (%)', fontweight='bold')
    ax.set_title('Practical Research Indicators: Attack vs Defense Papers\n(Higher = More Practical)',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(indicators)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_attack_vs_defense.png')
    plt.savefig(OUTPUT_DIR / 'fig10_attack_vs_defense.pdf')
    plt.close()
    print("  Saved: fig10_attack_vs_defense.png/pdf")

# ============================================================================
# BONUS FIGURE 11: Correlation Heatmap of Gap Flags
# ============================================================================

def figure_11_flag_correlation(df):
    """Create correlation heatmap of the 6 gap flags."""
    print("\nGenerating Figure 11 (Bonus): Gap Flag Correlations...")
    
    flag_columns = ['Flag_Grad', 'Flag_HighQ', 'Flag_WB', 'Flag_NoEcon', 'Flag_NoCode', 'Flag_NoReal']
    flag_labels = ['Requires\nGradients', 'High Query\nBudget', 'White-box\nOnly', 
                   'No\nEconomics', 'No Code\nReleased', 'No Real\nSystem']
    
    # Calculate correlation matrix
    corr_matrix = df[flag_columns].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom", fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(len(flag_labels)))
    ax.set_yticks(np.arange(len(flag_labels)))
    ax.set_xticklabels(flag_labels, fontsize=9)
    ax.set_yticklabels(flag_labels, fontsize=9)
    
    # Add text annotations
    for i in range(len(flag_labels)):
        for j in range(len(flag_labels)):
            val = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_title('Correlation Between Gap Indicators\n(Do Problems Cluster Together?)',
                 fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_flag_correlation.png')
    plt.savefig(OUTPUT_DIR / 'fig11_flag_correlation.pdf')
    plt.close()
    print("  Saved: fig11_flag_correlation.png/pdf")

# ============================================================================
# BONUS FIGURE 12: Gap Score Distribution Histogram
# ============================================================================

def figure_12_gap_score_histogram(df):
    """Create histogram of Traditional_Score distribution."""
    print("\nGenerating Figure 12 (Bonus): Gap Score Distribution...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    scores = df['Traditional_Score'].values
    bins = np.arange(-0.5, 7.5, 1)  # 0 to 6
    
    n, bins_out, patches = ax.hist(scores, bins=bins, edgecolor='white', linewidth=1.5)
    
    # Color bars by gap severity (gradient from green to red)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    # Add count labels
    for i, (count, patch) in enumerate(zip(n, patches)):
        ax.text(patch.get_x() + patch.get_width()/2, count + 2,
               f'{int(count)}\n({count/len(df)*100:.1f}%)', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add mean line
    mean_score = df['Traditional_Score'].mean()
    ax.axvline(x=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    
    # Formatting
    ax.set_xlabel('Traditional Score (Gap Score)', fontweight='bold')
    ax.set_ylabel('Number of Papers', fontweight='bold')
    ax.set_title(f'Distribution of Theory-Practice Gap Scores (n={len(df)})\n(0 = Most Practical, 6 = Most Academic)',
                 fontweight='bold', pad=15)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['0\n(Best)', '1', '2', '3', '4', '5', '6\n(Worst)'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(n) * 1.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_gap_score_histogram.png')
    plt.savefig(OUTPUT_DIR / 'fig12_gap_score_histogram.pdf')
    plt.close()
    print("  Saved: fig12_gap_score_histogram.png/pdf")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to generate all figures and statistics."""
    print("="*70)
    print("ADVERSARIAL ML RESEARCH-PRACTICE GAP ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data(INPUT_CSV)
    
    # Calculate and print statistics
    calculate_all_statistics(df)
    
    # Generate all figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    figure_1_dataset_overview(df)
    figure_2_overall_gap(df)
    figure_3_gap_by_conference_heatmap(df)
    figure_4_practical_trends(df)
    figure_5_threat_model_distribution(df)
    figure_6_gradient_by_attack_type(df)
    figure_7_domain_distribution(df)
    figure_8_gap_score_by_conference(df)
    figure_9_gap_score_trend(df)
    figure_10_attack_vs_defense(df)
    figure_11_flag_correlation(df)
    figure_12_gap_score_histogram(df)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()