"""
Generate LaTeX table code from CSV for appendix
"""

import pandas as pd

def generate_latex_tables(csv_path, output_path='/mnt/user-data/outputs/'):
    """Generate LaTeX table code from the CSV data"""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Clean up title field - remove extra quotes and truncate long titles
    df['Title'] = df['Title'].fillna('').str.replace('"', '').str[:50]
    
    # Replace NaN with empty strings
    df = df.fillna('')
    
    # Generate Part 1: Basic Info & Categories
    with open(f'{output_path}appendix_table_part1.tex', 'w') as f:
        f.write("\\begin{landscape}\n")
        f.write("\\begin{longtable}{|p{0.8cm}|p{2.2cm}|p{4.5cm}|p{0.7cm}|p{1.1cm}|p{0.8cm}|p{1cm}|p{0.6cm}|p{0.6cm}|p{0.6cm}|}\n")
        f.write("\\caption{Complete Analysis of ACM CCS 2022-2025 Papers (Part 1: Basic Info \\& Categories)} \\\\\n")
        f.write("\\hline\n")
        f.write("\\textbf{Year} & \\textbf{Filename} & \\textbf{Title} & \\textbf{G1} & \\textbf{G2} & \\textbf{G3} & \\textbf{G4} & \\textbf{G5} & \\textbf{G6} & \\textbf{G7} \\\\\n")
        f.write("\\hline\n")
        f.write("\\endfirsthead\n\n")
        
        f.write("\\multicolumn{10}{c}%\n")
        f.write("{{\\bfseries Table \\thetable\\ continued from previous page}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\textbf{Year} & \\textbf{Filename} & \\textbf{Title} & \\textbf{G1} & \\textbf{G2} & \\textbf{G3} & \\textbf{G4} & \\textbf{G5} & \\textbf{G6} & \\textbf{G7} \\\\\n")
        f.write("\\hline\n")
        f.write("\\endhead\n\n")
        
        f.write("\\hline \\multicolumn{10}{|r|}{{Continued on next page}} \\\\ \\hline\n")
        f.write("\\endfoot\n\n")
        
        f.write("\\hline\n")
        f.write("\\endlastfoot\n\n")
        
        # Write data rows
        for _, row in df.iterrows():
            # Escape special LaTeX characters
            filename = str(row['Filename']).replace('_', '\\_').replace('.pdf', '')
            title = str(row['Title']).replace('&', '\\&').replace('_', '\\_').replace('#', '\\#')
            
            f.write(f"{row['Year']} & {filename} & {title} & {row['G1']} & {row['G2']} & {row['G3']} & {row['G4']} & {row['G5']} & {row['G6']} & {row['G7']} \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{longtable}\n")
        f.write("\\end{landscape}\n")
    
    # Generate Part 2: Threat Model & Metrics
    with open(f'{output_path}appendix_table_part2.tex', 'w') as f:
        f.write("\\begin{landscape}\n")
        f.write("\\begin{longtable}{|p{0.8cm}|p{2.2cm}|p{1.3cm}|p{1cm}|p{0.7cm}|p{0.7cm}|p{0.7cm}|p{0.8cm}|p{0.8cm}|p{0.7cm}|p{0.8cm}|p{0.8cm}|p{0.8cm}|p{0.9cm}|}\n")
        f.write("\\caption{Complete Analysis of ACM CCS 2022-2025 Papers (Part 2: Threat Model \\& Metrics)} \\\\\n")
        f.write("\\hline\n")
        f.write("\\textbf{Year} & \\textbf{Filename} & \\textbf{T1} & \\textbf{T2} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & \\textbf{FGrd} & \\textbf{FHiQ} & \\textbf{FWB} & \\textbf{FNoEc} & \\textbf{FNoCd} & \\textbf{FNoRl} & \\textbf{Score} \\\\\n")
        f.write("\\hline\n")
        f.write("\\endfirsthead\n\n")
        
        f.write("\\multicolumn{14}{c}%\n")
        f.write("{{\\bfseries Table \\thetable\\ continued from previous page}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\textbf{Year} & \\textbf{Filename} & \\textbf{T1} & \\textbf{T2} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & \\textbf{FGrd} & \\textbf{FHiQ} & \\textbf{FWB} & \\textbf{FNoEc} & \\textbf{FNoCd} & \\textbf{FNoRl} & \\textbf{Score} \\\\\n")
        f.write("\\hline\n")
        f.write("\\endhead\n\n")
        
        f.write("\\hline \\multicolumn{14}{|r|}{{Continued on next page}} \\\\ \\hline\n")
        f.write("\\endfoot\n\n")
        
        f.write("\\hline\n")
        f.write("\\endlastfoot\n\n")
        
        # Write data rows
        for _, row in df.iterrows():
            filename = str(row['Filename']).replace('_', '\\_').replace('.pdf', '')
            
            f.write(f"{row['Year']} & {filename} & {row['T1']} & {row['T2']} & {row['Q1']} & {row['Q2']} & {row['Q3']} & ")
            f.write(f"{row['Flag_Grad']} & {row['Flag_HighQ']} & {row['Flag_WB']} & {row['Flag_NoEcon']} & ")
            f.write(f"{row['Flag_NoCode']} & {row['Flag_NoReal']} & {row['Traditional_Score']} \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{longtable}\n")
        f.write("\\end{landscape}\n")
    
    print("✓ Generated LaTeX table code for appendix")
    print("  - appendix_table_part1.tex (Basic Info & Categories)")
    print("  - appendix_table_part2.tex (Threat Model & Metrics)")

def main():
    generate_latex_tables('analysis_results_clean.csv')

if __name__ == "__main__":
    main()