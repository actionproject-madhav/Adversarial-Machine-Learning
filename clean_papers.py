#!/usr/bin/env python3
"""
Clean the extracted papers list by flagging non-AML papers.
"""

import csv
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "papers_with_metadata.csv"
OUTPUT_CSV = Path(__file__).parent / "papers_cleaned.csv"

# Papers to EXCLUDE (not adversarial ML - just dependencies/utilities)
EXCLUDE_ARXIV_IDS = {
    '1412.6980',  # Adam optimizer - not AML
    '1506.02025', # Spatial Transformer Networks - not AML
    '1611.05431', # ResNeXt architecture - not AML
    '2111.14725', # Vision Transformer search - not AML
    '2501.01151', # Oscillations in frequency - definitely not AML (wrong paper)
    '2505.21605', # SOSBENCH - too recent, may not exist yet
    '2403.12025', # Health equity toolbox - not AML
    '2406.18682', # Multilingual Alignment Prism - not AML
}

# Papers that are borderline (defenses, not attacks) - keep but flag
BORDERLINE_PAPERS = {
    '1507.00677',  # Virtual Adversarial Training - defense
    '1905.11268',  # Robust Word Recognition - defense
    '1909.06723',  # Synonym-based defense - defense
}

def main():
    papers = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            papers.append(row)
    
    # Add is_aml_paper column
    new_fieldnames = ['is_aml_paper'] + list(fieldnames)
    
    cleaned = []
    excluded_count = 0
    
    for paper in papers:
        arxiv_id = paper['arxiv_id']
        
        if arxiv_id in EXCLUDE_ARXIV_IDS:
            paper['is_aml_paper'] = 'NO - Exclude'
            excluded_count += 1
        elif arxiv_id in BORDERLINE_PAPERS:
            paper['is_aml_paper'] = 'YES - Defense'
        else:
            paper['is_aml_paper'] = 'YES'
        
        cleaned.append(paper)
    
    # Write cleaned output
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for paper in cleaned:
            writer.writerow(paper)
    
    aml_count = sum(1 for p in cleaned if p['is_aml_paper'].startswith('YES'))
    
    print(f"Total papers: {len(cleaned)}")
    print(f"AML papers (to code): {aml_count}")
    print(f"Excluded (not AML): {excluded_count}")
    print(f"\nOutput: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
