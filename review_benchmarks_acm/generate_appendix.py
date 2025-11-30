"""
Generate Appendix Tables as High-Quality Images
Creates publication-quality table images from CSV data
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path

# Get script directory and set up paths
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "analysis_results_clean.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 7
plt.rcParams['figure.dpi'] = 300

def create_table_image_part1(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Generate Part 1: Basic Info & Categories as image
    Columns: Year, Filename, Title, G1-G7
    """
    # Select relevant columns
    cols = ['Year', 'Filename', 'Title', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
    table_data = df[cols].copy()
    
    # Truncate long text
    table_data['Filename'] = table_data['Filename'].str.replace('.pdf', '').str[:15]
    table_data['Title'] = table_data['Title'].fillna('').str[:35] + '...'
    table_data = table_data.fillna('')
    
    # Convert to list of lists for table
    data = table_data.values.tolist()
    
    # Create figure - split into multiple pages if needed
    rows_per_page = 35
    n_pages = (len(data) + rows_per_page - 1) // rows_per_page
    
    for page in range(n_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(data))
        page_data = data[start_idx:end_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=page_data,
                        colLabels=['Year', 'Filename', 'Title', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.06, 0.14, 0.28, 0.06, 0.09, 0.06, 0.08, 0.06, 0.06, 0.06])
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(10):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(page_data) + 1):
            for j in range(10):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('#FFFFFF')
                cell.set_edgecolor('#CCCCCC')
        
        # Add title
        title = f'Complete Analysis of ACM CCS 2022-2025 Papers\nPart 1: Basic Info & Categories'
        if n_pages > 1:
            title += f' (Page {page + 1} of {n_pages})'
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add footer with legend
        footer_text = (
            'G1=Type (atk/def/both), G2=Attack Type, G3=ML Type (DL/Traditional/Both), G4=Domain, '
            'G5=Real-world Testing, G6=Economic Considerations, G7=Code Released'
        )
        plt.figtext(0.5, 0.02, footer_text, ha='center', fontsize=6, style='italic')
        
        plt.tight_layout()
        
        # Save
        if n_pages == 1:
            filename = f'{output_path}appendix_table_part1.png'
        else:
            filename = f'{output_path}appendix_table_part1_page{page + 1}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Generated {filename}")
        plt.close()

def create_table_image_part2(df, output_path=None):
    if output_path is None:
        output_path = str(OUTPUT_DIR) + '/'
    """
    Generate Part 2: Threat Model & Metrics as image
    Columns: Year, Filename, T1, T2, Q1-Q3, Flags, Score
    """
    # Select relevant columns
    cols = ['Year', 'Filename', 'T1', 'T2', 'Q1', 'Q2', 'Q3', 
            'Flag_Grad', 'Flag_HighQ', 'Flag_WB', 'Flag_NoEcon', 'Flag_NoCode', 'Flag_NoReal', 'Traditional_Score']
    table_data = df[cols].copy()
    
    # Truncate filename
    table_data['Filename'] = table_data['Filename'].str.replace('.pdf', '').str[:15]
    table_data = table_data.fillna('')
    
    # Convert to list of lists
    data = table_data.values.tolist()
    
    # Create figure - split into multiple pages
    rows_per_page = 35
    n_pages = (len(data) + rows_per_page - 1) // rows_per_page
    
    for page in range(n_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(data))
        page_data = data[start_idx:end_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        col_labels = ['Year', 'File', 'T1', 'T2', 'Q1', 'Q2', 'Q3', 
                     'FGrd', 'FHiQ', 'FWB', 'FNoEc', 'FNoCd', 'FNoRl', 'Score']
        table = ax.table(cellText=page_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.06, 0.12, 0.08, 0.07, 0.06, 0.06, 0.06,
                                 0.06, 0.06, 0.06, 0.07, 0.07, 0.07, 0.07])
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(14):
            cell = table[(0, i)]
            cell.set_facecolor('#A23B72')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors and highlight high scores
        for i in range(1, len(page_data) + 1):
            score = page_data[i-1][13]  # Last column is score
            for j in range(14):
                cell = table[(i, j)]
                
                # Color based on score in last column
                if j == 13:  # Score column
                    try:
                        score_val = int(score) if score != '' else 0
                        if score_val >= 4:
                            cell.set_facecolor('#FFE5E5')  # Light red for high concern
                        elif score_val >= 2:
                            cell.set_facecolor('#FFF5E5')  # Light orange for medium
                        else:
                            cell.set_facecolor('#E5F5E5')  # Light green for low
                    except:
                        cell.set_facecolor('#FFFFFF')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#F0F0F0')
                    else:
                        cell.set_facecolor('#FFFFFF')
                
                cell.set_edgecolor('#CCCCCC')
        
        # Add title
        title = f'Complete Analysis of ACM CCS 2022-2025 Papers\nPart 2: Threat Model & Metrics'
        if n_pages > 1:
            title += f' (Page {page + 1} of {n_pages})'
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add footer with legend
        footer_text = (
            'T1=Threat Model, T2=Knowledge Level, Q1=Gradient-based, Q2=Query Budget, Q3=Query Complexity | '
            'Flags: FGrd=Gradient, FHiQ=High Query, FWB=White-Box, FNoEc=No Economics, FNoCd=No Code, FNoRl=No Real-World | '
            'Score: 0-6 (higher = more concerning from practical deployment perspective)'
        )
        plt.figtext(0.5, 0.02, footer_text, ha='center', fontsize=6, style='italic', wrap=True)
        
        plt.tight_layout()
        
        # Save
        if n_pages == 1:
            filename = f'{output_path}appendix_table_part2.png'
        else:
            filename = f'{output_path}appendix_table_part2_page{page + 1}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Generated {filename}")
        plt.close()

def create_summary_table_image(df, output_path='/mnt/user-data/outputs/'):
    """
    Generate summary statistics table as image
    """
    # Calculate statistics by year
    stats_by_year = []
    
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        
        stats = {
            'Year': year,
            'Total': len(year_data),
            'Attack': (year_data['G1'] == 'atk').sum(),
            'Defense': (year_data['G1'] == 'def').sum(),
            'Both': (year_data['G1'] == 'both').sum(),
            'Grad%': f"{(year_data['Flag_Grad'] == 1).sum() / len(year_data) * 100:.1f}",
            'HighQ%': f"{(year_data['Flag_HighQ'] == 1).sum() / len(year_data) * 100:.1f}",
            'WB%': f"{(year_data['Flag_WB'] == 1).sum() / len(year_data) * 100:.1f}",
            'BB%': f"{(year_data['T1'] == 'Black-box').sum() / len(year_data) * 100:.1f}",
            'Real%': f"{(year_data['Flag_NoReal'] == 0).sum() / len(year_data) * 100:.1f}",
            'Code%': f"{(year_data['Flag_NoCode'] == 0).sum() / len(year_data) * 100:.1f}",
            'Econ%': f"{(year_data['Flag_NoEcon'] == 0).sum() / len(year_data) * 100:.1f}",
        }
        stats_by_year.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_by_year)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    col_labels = ['Year', 'Total\nPapers', 'Attack\nPapers', 'Defense\nPapers', 'Both\nPapers',
                 'Gradient\nBased %', 'High Query\nBudget %', 'White-Box\n%', 'Black-Box\n%',
                 'Real-World\nTesting %', 'Code\nReleased %', 'Economic\nConsider. %']
    
    data = stats_df.values.tolist()
    
    # Create table
    table = ax.table(cellText=data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.08, 0.08, 0.09, 0.09, 0.09])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(12):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(data) + 1):
        for j in range(12):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#CCCCCC')
            
            # Bold year column
            if j == 0:
                cell.set_text_props(weight='bold')
    
    # Add title
    plt.title('Summary Statistics by Year (2022-2025)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    filename = f'{output_path}appendix_summary_table.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Generated {filename}")
    plt.close()

def main():
    """Main function to generate all appendix table images"""
    print("\n" + "="*60)
    print("Generating Appendix Tables as Images")
    print("="*60 + "\n")
    
    # Load data
    df = pd.read_csv(str(CSV_FILE))
    print(f"Loaded {len(df)} papers\n")
    
    # Generate tables
    print("Generating Part 1 (Basic Info & Categories)...")
    create_table_image_part1(df)
    
    print("\nGenerating Part 2 (Threat Model & Metrics)...")
    create_table_image_part2(df)
    
    print("\nGenerating Summary Statistics Table...")
    create_summary_table_image(df)
    
    print("\n" + "="*60)
    print("All appendix table images generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  • appendix_table_part1_page1.png/.pdf (and possibly more pages)")
    print("  • appendix_table_part2_page1.png/.pdf (and possibly more pages)")
    print("  • appendix_summary_table.png/.pdf")
    print("\nUpload the PDF versions to Overleaf for best quality!\n")

if __name__ == "__main__":
    main()