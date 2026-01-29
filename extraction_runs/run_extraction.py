#!/usr/bin/env python3
"""
Robust paper extraction script - v2.
Extracts papers from artifact repos with comprehensive pattern matching.
Usage: python run_extraction.py run1|run2|compare
"""

import csv
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
REPO_DIR = BASE_DIR / "artifact_repos"

# ============================================================================
# COMPREHENSIVE ARXIV PATTERNS
# ============================================================================
ARXIV_PATTERNS = [
    # Standard new format: arxiv.org/abs/XXXX.XXXXX
    re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', re.IGNORECASE),
    # With http/https
    re.compile(r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5})', re.IGNORECASE),
    # arXiv: XXXX.XXXXX or arXiv:XXXX.XXXXX
    re.compile(r'arXiv[:\s]+(\d{4}\.\d{4,5})', re.IGNORECASE),
    # arxiv:XXXX.XXXXX (lowercase)
    re.compile(r'\barxiv[:\s]+(\d{4}\.\d{4,5})', re.IGNORECASE),
    # PDF links: arxiv.org/pdf/XXXX.XXXXX
    re.compile(r'arxiv\.org/pdf/(\d{4}\.\d{4,5})', re.IGNORECASE),
    # In BibTeX: eprint = {XXXX.XXXXX}
    re.compile(r'eprint\s*=\s*[{\"](\d{4}\.\d{4,5})[}\"]', re.IGNORECASE),
    # Old format: arxiv.org/abs/cs/XXXXXXX (pre-2007)
    re.compile(r'arxiv\.org/abs/([a-z-]+/\d{7})', re.IGNORECASE),
]

# File extensions to search
FILE_EXTENSIONS = ['.py', '.md', '.rst', '.txt', '.json', '.yaml', '.yml', '.bib', '.cfg']

# All artifacts with their repos
ARTIFACTS = [
    # Tools (≥1,000 GitHub stars)
    ('cleverhans', 'CleverHans', 'Tool'),           # 6,401 stars
    ('adversarial-robustness-toolbox', 'IBM ART', 'Tool'),  # 5,789 stars
    ('foolbox', 'Foolbox', 'Tool'),                 # 2,936 stars
    ('TextAttack', 'TextAttack', 'Tool'),           # 3,348 stars
    ('PyRIT', 'PyRIT', 'Tool'),                     # 3,343 stars
    # Benchmarks (peer-reviewed publications)
    ('robustbench', 'RobustBench', 'Benchmark'),    # NeurIPS 2021
    ('HarmBench', 'HarmBench', 'Benchmark'),        # ICML 2024
    ('auto-attack', 'AutoAttack', 'Benchmark'),     # ICML 2020
    # Regulatory (MITRE ATLAS)
    ('atlas-data', 'MITRE ATLAS', 'Regulatory'),    # Industry threat framework
]

ARTIFACT_TO_REPO = {art[1]: art[0] for art in ARTIFACTS}

# Artifact metadata for documentation
ARTIFACT_METADATA = {
    'CleverHans': {'stars': 6401, 'criterion': '≥1,000 stars'},
    'IBM ART': {'stars': 5789, 'criterion': '≥1,000 stars'},
    'Foolbox': {'stars': 2936, 'criterion': '≥1,000 stars'},
    'TextAttack': {'stars': 3348, 'criterion': '≥1,000 stars'},
    'PyRIT': {'stars': 3343, 'criterion': '≥1,000 stars'},
    'RobustBench': {'stars': 761, 'criterion': 'NeurIPS 2021'},
    'HarmBench': {'stars': 841, 'criterion': 'ICML 2024'},
    'AutoAttack': {'stars': 734, 'criterion': 'ICML 2020'},
    'MITRE ATLAS': {'stars': 'N/A', 'criterion': 'Industry threat framework'},
}

# Non-AML papers to exclude (optimizers, architectures, unrelated)
EXCLUDE_ARXIV_IDS = {
    '1412.6980',  # Adam optimizer
    '1506.02025', # Spatial Transformer Networks
    '1611.05431', # ResNeXt architecture
    '2111.14725', # Vision Transformer search
    '2501.01151', # Physics paper (wrong match)
    '2505.21605', # Future paper
    '2403.12025', # Health equity toolbox
    '2406.18682', # Multilingual alignment
    '1409.1556',  # VGGNet
    '1512.03385', # ResNet
    '1409.4842',  # GoogLeNet
    '1502.03167', # Batch normalization
}

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_arxiv_ids(filepath):
    """Extract all arXiv IDs from a file using comprehensive patterns."""
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except:
        return []
    
    found_ids = set()
    for pattern in ARXIV_PATTERNS:
        for match in pattern.finditer(content):
            arxiv_id = match.group(1)
            # Normalize old format
            if '/' in arxiv_id:
                continue  # Skip old format for now (rare)
            found_ids.add(arxiv_id)
    
    return list(found_ids)

def scan_repo(repo_name, artifact_name, artifact_type):
    """Scan a repository for paper references."""
    repo_path = REPO_DIR / repo_name
    if not repo_path.exists():
        print(f"    WARNING: {repo_name} not found!")
        return []
    
    results = []
    
    # Scan all relevant file types
    for ext in FILE_EXTENSIONS:
        for filepath in repo_path.rglob(f"*{ext}"):
            # Skip test files, examples, etc.
            rel_path = str(filepath.relative_to(repo_path))
            if 'test' in rel_path.lower() and 'attack' not in rel_path.lower():
                continue
            
            arxiv_ids = extract_arxiv_ids(filepath)
            for arxiv_id in arxiv_ids:
                results.append({
                    'artifact': artifact_name,
                    'artifact_type': artifact_type,
                    'arxiv_id': arxiv_id,
                    'source_file': rel_path
                })
    
    return results

def fetch_arxiv_metadata(arxiv_id):
    """Fetch metadata from arXiv API with retry."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            root = ET.fromstring(data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            
            if entry is None:
                return None
            
            # Check if paper exists (not just an ID error)
            id_elem = entry.find('atom:id', ns)
            if id_elem is None or 'Error' in (id_elem.text or ''):
                return None
            
            title = entry.find('atom:title', ns)
            title = title.text.strip().replace('\n', ' ') if title is not None else ""
            
            # Skip if title indicates an error
            if not title or 'Error' in title:
                return None
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            published = entry.find('atom:published', ns)
            pub_date = published.text[:10] if published is not None else ""
            
            return {
                'title': title,
                'authors': '; '.join(authors[:5]) + ('...' if len(authors) > 5 else ''),
                'pub_date': pub_date
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            continue
    
    return None

def get_first_commit_date_for_string(repo_path, search_string):
    """Get first commit date when a specific string was added to the repo."""
    try:
        # Use -S to find when the string was first introduced
        result = subprocess.run(
            ['git', 'log', '--all', '--format=%aI', '-S', search_string],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            dates = result.stdout.strip().split('\n')
            # Get the earliest (last in list) date when string was introduced
            oldest_date = dates[-1][:10]
            # Sanity check: date should be in the past
            if oldest_date <= datetime.now().strftime('%Y-%m-%d'):
                return oldest_date
    except:
        pass
    return None

def run_extraction(output_dir):
    """Run the complete extraction pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"=== EXTRACTION TO {output_dir.name} ===\n")
    
    # Step 1: Extract references from repos
    print("Step 1: Scanning repositories...")
    all_refs = []
    
    for repo_name, artifact_name, artifact_type in ARTIFACTS:
        refs = scan_repo(repo_name, artifact_name, artifact_type)
        all_refs.extend(refs)
        print(f"  {artifact_name}: {len(refs)} references")
    
    # Deduplicate by arXiv ID
    arxiv_to_artifacts = defaultdict(set)
    arxiv_to_sources = defaultdict(list)
    
    for ref in all_refs:
        arxiv_id = ref['arxiv_id']
        arxiv_to_artifacts[arxiv_id].add(ref['artifact'])
        arxiv_to_sources[arxiv_id].append({
            'artifact': ref['artifact'],
            'source_file': ref['source_file']
        })
    
    unique_arxiv = sorted(arxiv_to_artifacts.keys())
    print(f"\nUnique arXiv papers found: {len(unique_arxiv)}")
    
    # Step 2: Fetch metadata from arXiv
    print("\nStep 2: Fetching arXiv metadata...")
    papers = []
    
    for i, arxiv_id in enumerate(unique_arxiv):
        print(f"  [{i+1}/{len(unique_arxiv)}] {arxiv_id}...", end=' ', flush=True)
        
        metadata = fetch_arxiv_metadata(arxiv_id)
        
        if metadata is None:
            print("SKIP (invalid)")
            continue
        
        # Determine if AML paper
        if arxiv_id in EXCLUDE_ARXIV_IDS:
            is_aml = 'NO - Exclude'
        else:
            is_aml = 'YES'
        
        paper = {
            'is_aml_paper': is_aml,
            'arxiv_id': arxiv_id,
            'found_in_artifacts': ', '.join(sorted(arxiv_to_artifacts[arxiv_id])),
            'num_artifacts': len(arxiv_to_artifacts[arxiv_id]),
            'paper_title': metadata['title'],
            'paper_authors': metadata['authors'],
            'paper_pub_date': metadata['pub_date'],
        }
        papers.append(paper)
        print("OK")
        
        time.sleep(0.3)  # Rate limit
    
    print(f"\nValid papers: {len(papers)}")
    
    # Step 3: Get adoption dates
    # Use -S search only for MITRE ATLAS (slow but accurate for aggregated files)
    # Use file-based approach for other repos (fast and accurate for individual files)
    print("\nStep 3: Getting adoption dates from git history...")
    
    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        sources = arxiv_to_sources.get(arxiv_id, [])
        artifacts = arxiv_to_artifacts.get(arxiv_id, set())
        
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(papers)}...")
        
        adoption_dates = []
        
        # For MITRE ATLAS, use -S search (more accurate for aggregated YAML files)
        if 'MITRE ATLAS' in artifacts:
            repo_path = REPO_DIR / 'atlas-data'
            date = get_first_commit_date_for_string(repo_path, arxiv_id)
            if date:
                adoption_dates.append({'artifact': 'MITRE ATLAS', 'date': date})
        
        # For other artifacts, use file-based approach
        for source in sources:
            artifact = source['artifact']
            if artifact == 'MITRE ATLAS':
                continue  # Already handled above
            source_file = source['source_file']
            repo_name = ARTIFACT_TO_REPO.get(artifact)
            
            if repo_name:
                repo_path = REPO_DIR / repo_name
                # Use file-based git log for most repos
                try:
                    result = subprocess.run(
                        ['git', 'log', '--follow', '--format=%aI', '--', source_file],
                        cwd=repo_path, capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        dates = result.stdout.strip().split('\n')
                        oldest_date = dates[-1][:10]
                        if oldest_date <= datetime.now().strftime('%Y-%m-%d'):
                            adoption_dates.append({'artifact': artifact, 'date': oldest_date})
                except:
                    pass
        
        if adoption_dates:
            # Filter out future dates
            valid_dates = [a for a in adoption_dates if a['date'] <= datetime.now().strftime('%Y-%m-%d')]
            if valid_dates:
                earliest = min(valid_dates, key=lambda x: x['date'])
                paper['first_adoption_date'] = earliest['date']
                paper['first_adoption_artifact'] = earliest['artifact']
                paper['all_adoptions'] = '; '.join([f"{a['artifact']}:{a['date']}" for a in valid_dates])
                
                # Calculate lag
                if paper['paper_pub_date']:
                    try:
                        pub = datetime.strptime(paper['paper_pub_date'], '%Y-%m-%d')
                        adopt = datetime.strptime(earliest['date'], '%Y-%m-%d')
                        lag_months = round((adopt - pub).days / 30.44, 1)
                        paper['adoption_lag_months'] = str(lag_months)
                    except:
                        paper['adoption_lag_months'] = ''
                else:
                    paper['adoption_lag_months'] = ''
            else:
                paper['first_adoption_date'] = ''
                paper['first_adoption_artifact'] = ''
                paper['all_adoptions'] = ''
                paper['adoption_lag_months'] = ''
        else:
            paper['first_adoption_date'] = ''
            paper['first_adoption_artifact'] = ''
            paper['all_adoptions'] = ''
            paper['adoption_lag_months'] = ''
    
    # Step 4: Write outputs
    print("\nStep 4: Writing outputs...")
    
    fieldnames = [
        'is_aml_paper', 'arxiv_id', 'found_in_artifacts', 'num_artifacts',
        'paper_title', 'paper_authors', 'paper_pub_date', 
        'first_adoption_date', 'first_adoption_artifact', 'all_adoptions', 
        'adoption_lag_months',
        'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1'
    ]
    
    # Add empty coding columns
    for paper in papers:
        for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
            paper[col] = ''
    
    # Write full dataset
    output_file = output_dir / 'papers_all.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)
    
    # Write grouped by artifact
    output_grouped = output_dir / 'papers_by_artifact.csv'
    papers_sorted = sorted(papers, key=lambda x: (x['found_in_artifacts'], x['arxiv_id']))
    with open(output_grouped, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers_sorted)
    
    # Summary
    aml_papers = [p for p in papers if p['is_aml_paper'] == 'YES']
    multi_artifact = [p for p in aml_papers if p['num_artifacts'] >= 2]
    
    print(f"\n=== SUMMARY ===")
    print(f"Total valid papers: {len(papers)}")
    print(f"AML papers: {len(aml_papers)}")
    print(f"Papers in 2+ artifacts: {len(multi_artifact)}")
    
    # By artifact breakdown
    print(f"\n=== BY ARTIFACT ===")
    for _, artifact_name, _ in ARTIFACTS:
        count = sum(1 for p in aml_papers if artifact_name in p['found_in_artifacts'])
        print(f"  {artifact_name}: {count} papers")
    
    print(f"\nOutput files:")
    print(f"  {output_file}")
    print(f"  {output_grouped}")
    
    return papers

def compare_runs(run1_dir, run2_dir):
    """Compare two extraction runs."""
    print("\n=== COMPARING RUNS ===\n")
    
    def load_run(run_dir):
        papers = {}
        csv_file = run_dir / 'papers_all.csv'
        if not csv_file.exists():
            csv_file = run_dir / 'papers_extracted.csv'  # Fallback
        with open(csv_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                papers[row['arxiv_id']] = row
        return papers
    
    run1 = load_run(Path(run1_dir))
    run2 = load_run(Path(run2_dir))
    
    all_ids = set(run1.keys()) | set(run2.keys())
    
    matches = 0
    mismatches = []
    
    fields_to_compare = ['paper_title', 'paper_pub_date', 'found_in_artifacts', 
                         'first_adoption_date', 'first_adoption_artifact']
    
    for arxiv_id in all_ids:
        if arxiv_id not in run1:
            mismatches.append(f"{arxiv_id}: Missing in run1")
            continue
        if arxiv_id not in run2:
            mismatches.append(f"{arxiv_id}: Missing in run2")
            continue
        
        paper1 = run1[arxiv_id]
        paper2 = run2[arxiv_id]
        
        paper_match = True
        for field in fields_to_compare:
            if paper1.get(field, '') != paper2.get(field, ''):
                paper_match = False
                mismatches.append(f"{arxiv_id}.{field}: differs")
        
        if paper_match:
            matches += 1
    
    total = len(all_ids)
    agreement = matches / total * 100 if total > 0 else 0
    
    print(f"Total papers: {total}")
    print(f"Perfect matches: {matches}")
    print(f"Agreement rate: {agreement:.1f}%")
    
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches[:10]:
            print(f"  {m}")
    
    return agreement

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_extraction.py run1|run2|compare")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    
    if sys.argv[1] == 'compare':
        compare_runs(script_dir / 'run1', script_dir / 'run2')
    else:
        run_name = sys.argv[1]
        output_dir = script_dir / run_name
        run_extraction(output_dir)
