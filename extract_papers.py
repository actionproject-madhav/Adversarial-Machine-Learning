#!/usr/bin/env python3
"""
Extract paper references from adversarial ML tool repositories.
Searches for arXiv links, DOIs, paper titles, and author citations in source code.
"""

import os
import re
import csv
import subprocess
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent / "artifact_repos"
OUTPUT_CSV = Path(__file__).parent / "paper_extraction_results.csv"

# Patterns to find paper references
ARXIV_PATTERN = re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', re.IGNORECASE)
ARXIV_PATTERN2 = re.compile(r'arXiv[:\s]+(\d{4}\.\d{4,5})', re.IGNORECASE)
DOI_PATTERN = re.compile(r'doi\.org/(10\.\d{4,}/[^\s\)\"\']+)', re.IGNORECASE)

# Common paper title patterns in docstrings
PAPER_PATTERNS = [
    re.compile(r'["\']([^"\']{20,100})["\'].*(?:et al|arxiv|\d{4})', re.IGNORECASE),
    re.compile(r'Based on[:\s]+["\']?([^"\'\.]{20,100})["\']?', re.IGNORECASE),
    re.compile(r'Implements[:\s]+["\']?([^"\'\.]{20,100})["\']?', re.IGNORECASE),
    re.compile(r'Reference[:\s]+["\']?([^"\'\.]{20,100})["\']?', re.IGNORECASE),
]

# Author patterns (Name et al., Year)
AUTHOR_PATTERN = re.compile(r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+et\s+al\.?)?)[,\s]+(\d{4})', re.IGNORECASE)

def find_files(repo_path, extensions=('.py', '.md', '.rst', '.txt')):
    """Find all relevant files in a repository."""
    files = []
    for ext in extensions:
        files.extend(repo_path.rglob(f'*{ext}'))
    return files

def extract_from_file(filepath):
    """Extract paper references from a single file."""
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return []
    
    results = []
    
    # Find arXiv links
    for match in ARXIV_PATTERN.finditer(content):
        results.append({
            'type': 'arxiv',
            'value': match.group(1),
            'source_file': str(filepath)
        })
    for match in ARXIV_PATTERN2.finditer(content):
        results.append({
            'type': 'arxiv',
            'value': match.group(1),
            'source_file': str(filepath)
        })
    
    # Find DOIs
    for match in DOI_PATTERN.finditer(content):
        results.append({
            'type': 'doi',
            'value': match.group(1),
            'source_file': str(filepath)
        })
    
    # Find author citations
    for match in AUTHOR_PATTERN.finditer(content):
        results.append({
            'type': 'author_citation',
            'value': f"{match.group(1)}, {match.group(2)}",
            'source_file': str(filepath)
        })
    
    return results

def get_technique_from_filename(filepath):
    """Extract technique name from filename."""
    name = filepath.stem
    # Convert snake_case to readable name
    name = name.replace('_', ' ').title()
    # Skip common non-technique files
    skip_patterns = ['__init__', 'test', 'utils', 'base', 'common', 'config', 'setup']
    if any(p in name.lower() for p in skip_patterns):
        return None
    return name

def scan_cleverhans(repo_path):
    """Scan CleverHans for paper references."""
    results = []
    attacks_dir = repo_path / "cleverhans"
    
    if not attacks_dir.exists():
        return results
    
    for py_file in attacks_dir.rglob("*.py"):
        technique = get_technique_from_filename(py_file)
        if not technique:
            continue
        
        refs = extract_from_file(py_file)
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        
        # Look for specific attack docstrings
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        docstring = docstring_match.group(1) if docstring_match else ""
        
        results.append({
            'artifact': 'CleverHans',
            'artifact_type': 'Tool',
            'technique': technique,
            'source_file': str(py_file.relative_to(repo_path)),
            'references': refs,
            'docstring_snippet': docstring[:500] if docstring else ""
        })
    
    return results

def scan_art(repo_path):
    """Scan IBM ART for paper references."""
    results = []
    attacks_dir = repo_path / "art" / "attacks"
    
    if not attacks_dir.exists():
        return results
    
    for py_file in attacks_dir.rglob("*.py"):
        technique = get_technique_from_filename(py_file)
        if not technique:
            continue
        
        refs = extract_from_file(py_file)
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        
        # ART often has | Paper link: in docstrings
        paper_links = re.findall(r'\|\s*Paper link[:\s]+([^\n|]+)', content)
        
        results.append({
            'artifact': 'IBM ART',
            'artifact_type': 'Tool',
            'technique': technique,
            'source_file': str(py_file.relative_to(repo_path)),
            'references': refs,
            'paper_links': paper_links
        })
    
    return results

def scan_foolbox(repo_path):
    """Scan Foolbox for paper references."""
    results = []
    attacks_dir = repo_path / "foolbox" / "attacks"
    
    if not attacks_dir.exists():
        return results
    
    for py_file in attacks_dir.rglob("*.py"):
        technique = get_technique_from_filename(py_file)
        if not technique:
            continue
        
        refs = extract_from_file(py_file)
        
        results.append({
            'artifact': 'Foolbox',
            'artifact_type': 'Tool',
            'technique': technique,
            'source_file': str(py_file.relative_to(repo_path)),
            'references': refs
        })
    
    return results

def scan_textattack(repo_path):
    """Scan TextAttack for paper references."""
    results = []
    attacks_dir = repo_path / "textattack" / "attack_recipes"
    
    if not attacks_dir.exists():
        return results
    
    for py_file in attacks_dir.rglob("*.py"):
        technique = get_technique_from_filename(py_file)
        if not technique:
            continue
        
        refs = extract_from_file(py_file)
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        
        results.append({
            'artifact': 'TextAttack',
            'artifact_type': 'Tool',
            'technique': technique,
            'source_file': str(py_file.relative_to(repo_path)),
            'references': refs
        })
    
    return results

def scan_robustbench(repo_path):
    """Scan RobustBench for paper references - focus on model_info."""
    results = []
    
    # RobustBench has model_info directory with YAML files
    model_info_dir = repo_path / "robustbench" / "model_info"
    if model_info_dir.exists():
        for yaml_file in model_info_dir.rglob("*.json"):
            refs = extract_from_file(yaml_file)
            results.append({
                'artifact': 'RobustBench',
                'artifact_type': 'Benchmark',
                'technique': yaml_file.stem,
                'source_file': str(yaml_file.relative_to(repo_path)),
                'references': refs
            })
    
    # Also check Python files
    for py_file in repo_path.rglob("*.py"):
        refs = extract_from_file(py_file)
        if refs:
            results.append({
                'artifact': 'RobustBench',
                'artifact_type': 'Benchmark',
                'technique': 'Various',
                'source_file': str(py_file.relative_to(repo_path)),
                'references': refs
            })
    
    return results

def scan_harmbench(repo_path):
    """Scan HarmBench for paper references."""
    results = []
    
    # Check baselines directory
    baselines_dir = repo_path / "baselines"
    if baselines_dir.exists():
        for py_file in baselines_dir.rglob("*.py"):
            technique = get_technique_from_filename(py_file)
            if not technique:
                continue
            refs = extract_from_file(py_file)
            results.append({
                'artifact': 'HarmBench',
                'artifact_type': 'Benchmark',
                'technique': technique,
                'source_file': str(py_file.relative_to(repo_path)),
                'references': refs
            })
    
    # Also check README and paper references
    readme = repo_path / "README.md"
    if readme.exists():
        refs = extract_from_file(readme)
        if refs:
            results.append({
                'artifact': 'HarmBench',
                'artifact_type': 'Benchmark',
                'technique': 'README References',
                'source_file': 'README.md',
                'references': refs
            })
    
    return results

def scan_pyrit(repo_path):
    """Scan PyRIT for paper references."""
    results = []
    
    for py_file in repo_path.rglob("*.py"):
        refs = extract_from_file(py_file)
        if refs:
            technique = get_technique_from_filename(py_file)
            results.append({
                'artifact': 'PyRIT',
                'artifact_type': 'Tool',
                'technique': technique or py_file.stem,
                'source_file': str(py_file.relative_to(repo_path)),
                'references': refs
            })
    
    return results

def deduplicate_arxiv(all_results):
    """Deduplicate results by arXiv ID."""
    seen_arxiv = {}
    deduped = []
    
    for result in all_results:
        refs = result.get('references', [])
        arxiv_ids = [r['value'] for r in refs if r['type'] == 'arxiv']
        
        for arxiv_id in arxiv_ids:
            if arxiv_id not in seen_arxiv:
                seen_arxiv[arxiv_id] = result
                deduped.append({
                    'artifact_name': result['artifact'],
                    'artifact_type': result['artifact_type'],
                    'technique_name': result['technique'],
                    'arxiv_id': arxiv_id,
                    'source_file': result['source_file'],
                    'paper_title': '',  # To be filled manually or via API
                    'paper_authors': '',
                    'paper_pub_date': '',
                    'artifact_addition_date': '',
                    'adoption_lag_months': '',
                    'extraction_confidence': 'HIGH'
                })
    
    return deduped

def main():
    print("Scanning repositories for paper references...")
    
    all_results = []
    
    # Scan each repository
    repos = {
        'cleverhans': scan_cleverhans,
        'adversarial-robustness-toolbox': scan_art,
        'foolbox': scan_foolbox,
        'TextAttack': scan_textattack,
        'robustbench': scan_robustbench,
        'HarmBench': scan_harmbench,
        'PyRIT': scan_pyrit,
    }
    
    for repo_name, scan_func in repos.items():
        repo_path = REPO_DIR / repo_name
        if repo_path.exists():
            print(f"Scanning {repo_name}...")
            results = scan_func(repo_path)
            all_results.extend(results)
            print(f"  Found {len(results)} technique entries")
        else:
            print(f"  {repo_name} not found, skipping")
    
    # Collect all unique arXiv IDs
    all_arxiv = set()
    all_dois = set()
    all_authors = set()
    
    for result in all_results:
        for ref in result.get('references', []):
            if ref['type'] == 'arxiv':
                all_arxiv.add(ref['value'])
            elif ref['type'] == 'doi':
                all_dois.add(ref['value'])
            elif ref['type'] == 'author_citation':
                all_authors.add(ref['value'])
    
    print(f"\n=== SUMMARY ===")
    print(f"Total technique entries: {len(all_results)}")
    print(f"Unique arXiv IDs found: {len(all_arxiv)}")
    print(f"Unique DOIs found: {len(all_dois)}")
    print(f"Author citations found: {len(all_authors)}")
    
    # Write detailed results
    output_detailed = Path(__file__).parent / "extracted_references_detailed.csv"
    with open(output_detailed, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['artifact', 'artifact_type', 'technique', 'ref_type', 'ref_value', 'source_file'])
        for result in all_results:
            for ref in result.get('references', []):
                writer.writerow([
                    result['artifact'],
                    result['artifact_type'],
                    result['technique'],
                    ref['type'],
                    ref['value'],
                    result['source_file']
                ])
    
    # Write unique arXiv papers
    output_arxiv = Path(__file__).parent / "unique_arxiv_papers.csv"
    with open(output_arxiv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['arxiv_id', 'found_in_artifacts', 'paper_title', 'paper_authors', 'paper_pub_date', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'T1', 'T2', 'Q1', 'Q2', 'Q3'])
        
        # Track which artifacts each arXiv ID appears in
        arxiv_to_artifacts = {}
        for result in all_results:
            for ref in result.get('references', []):
                if ref['type'] == 'arxiv':
                    arxiv_id = ref['value']
                    if arxiv_id not in arxiv_to_artifacts:
                        arxiv_to_artifacts[arxiv_id] = set()
                    arxiv_to_artifacts[arxiv_id].add(result['artifact'])
        
        for arxiv_id in sorted(arxiv_to_artifacts.keys()):
            artifacts = ', '.join(sorted(arxiv_to_artifacts[arxiv_id]))
            writer.writerow([arxiv_id, artifacts, '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
    
    print(f"\nOutput files:")
    print(f"  - {output_detailed}")
    print(f"  - {output_arxiv}")
    
    # Print arXiv IDs for manual lookup
    print(f"\n=== ARXIV IDS TO LOOKUP ===")
    for arxiv_id in sorted(all_arxiv):
        print(f"https://arxiv.org/abs/{arxiv_id}")

if __name__ == "__main__":
    main()
