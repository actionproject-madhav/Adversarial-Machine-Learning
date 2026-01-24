#!/usr/bin/env python3
"""
Get adoption dates from git history for each paper.
"""

import csv
import subprocess
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "artifact_repos"
DETAILED_CSV = BASE_DIR / "extracted_references_detailed.csv"
CLEANED_CSV = BASE_DIR / "papers_cleaned.csv"
OUTPUT_CSV = BASE_DIR / "papers_with_adoption_dates.csv"

# Map artifact names to repo directories
ARTIFACT_TO_REPO = {
    'CleverHans': 'cleverhans',
    'IBM ART': 'adversarial-robustness-toolbox',
    'Foolbox': 'foolbox',
    'TextAttack': 'TextAttack',
    'RobustBench': 'robustbench',
    'HarmBench': 'HarmBench',
    'PyRIT': 'PyRIT',
}

def get_first_commit_date(repo_path, file_path):
    """Get the date of the first commit that added a file."""
    try:
        # Use git log to find the first commit for this file
        result = subprocess.run(
            ['git', 'log', '--follow', '--format=%aI', '--diff-filter=A', '--', file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            dates = result.stdout.strip().split('\n')
            # Return the oldest date (last in the list since git log shows newest first)
            # But --diff-filter=A shows when file was added, so it should be one date
            return dates[-1][:10]  # YYYY-MM-DD
        
        # Fallback: get the oldest commit that touched this file
        result = subprocess.run(
            ['git', 'log', '--follow', '--format=%aI', '--', file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            dates = result.stdout.strip().split('\n')
            return dates[-1][:10]  # Oldest date
            
    except Exception as e:
        print(f"    Error getting date for {file_path}: {e}")
    
    return None

def main():
    print("Reading detailed extraction data...")
    
    # Build map: arxiv_id -> [(artifact, source_file), ...]
    arxiv_to_sources = defaultdict(list)
    
    with open(DETAILED_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['ref_type'] == 'arxiv':
                arxiv_id = row['ref_value']
                artifact = row['artifact']
                source_file = row['source_file']
                arxiv_to_sources[arxiv_id].append({
                    'artifact': artifact,
                    'source_file': source_file
                })
    
    print(f"Found {len(arxiv_to_sources)} unique arXiv IDs with source files")
    
    # Get adoption dates for each arxiv_id
    arxiv_adoption_dates = {}
    
    for arxiv_id, sources in arxiv_to_sources.items():
        print(f"Processing {arxiv_id}...")
        
        adoption_info = []
        
        for source in sources:
            artifact = source['artifact']
            source_file = source['source_file']
            
            repo_name = ARTIFACT_TO_REPO.get(artifact)
            if not repo_name:
                continue
            
            repo_path = REPO_DIR / repo_name
            if not repo_path.exists():
                continue
            
            # Get the first commit date for this file
            date = get_first_commit_date(repo_path, source_file)
            
            if date:
                adoption_info.append({
                    'artifact': artifact,
                    'date': date,
                    'source_file': source_file
                })
                print(f"  {artifact}: {date}")
        
        if adoption_info:
            # Find the earliest adoption date
            earliest = min(adoption_info, key=lambda x: x['date'])
            arxiv_adoption_dates[arxiv_id] = {
                'first_adoption_date': earliest['date'],
                'first_adoption_artifact': earliest['artifact'],
                'all_adoptions': '; '.join([f"{a['artifact']}:{a['date']}" for a in adoption_info])
            }
    
    print(f"\nGot adoption dates for {len(arxiv_adoption_dates)} papers")
    
    # Read cleaned papers and add adoption dates
    papers = []
    with open(CLEANED_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            papers.append(row)
    
    # Add new columns
    new_fieldnames = fieldnames + ['first_adoption_date', 'first_adoption_artifact', 'all_adoptions', 'adoption_lag_months']
    
    for paper in papers:
        arxiv_id = paper['arxiv_id']
        adoption = arxiv_adoption_dates.get(arxiv_id, {})
        
        paper['first_adoption_date'] = adoption.get('first_adoption_date', '')
        paper['first_adoption_artifact'] = adoption.get('first_adoption_artifact', '')
        paper['all_adoptions'] = adoption.get('all_adoptions', '')
        
        # Calculate adoption lag
        pub_date = paper.get('paper_pub_date', '')
        adoption_date = paper.get('first_adoption_date', '')
        
        if pub_date and adoption_date:
            try:
                pub = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                adopt = datetime.strptime(adoption_date[:10], '%Y-%m-%d')
                lag_days = (adopt - pub).days
                lag_months = round(lag_days / 30.44, 1)  # Average days per month
                paper['adoption_lag_months'] = str(lag_months)
            except:
                paper['adoption_lag_months'] = ''
        else:
            paper['adoption_lag_months'] = ''
    
    # Write output
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper)
    
    print(f"\nOutput written to: {OUTPUT_CSV}")
    
    # Summary stats
    with_dates = sum(1 for p in papers if p.get('first_adoption_date'))
    with_lag = sum(1 for p in papers if p.get('adoption_lag_months'))
    
    print(f"Papers with adoption dates: {with_dates}")
    print(f"Papers with calculated lag: {with_lag}")
    
    # Show some sample adoption lags
    print("\n=== SAMPLE ADOPTION LAGS ===")
    sorted_papers = sorted([p for p in papers if p.get('adoption_lag_months')], 
                          key=lambda x: float(x['adoption_lag_months']))
    
    print("\nFastest adoptions:")
    for p in sorted_papers[:5]:
        print(f"  {p['adoption_lag_months']} months: {p['paper_title'][:50]}...")
    
    print("\nSlowest adoptions:")
    for p in sorted_papers[-5:]:
        print(f"  {p['adoption_lag_months']} months: {p['paper_title'][:50]}...")

if __name__ == "__main__":
    main()
