#!/usr/bin/env python3
"""
Fetch paper metadata from arXiv API for extracted paper IDs.
"""

import csv
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "unique_arxiv_papers.csv"
OUTPUT_CSV = Path(__file__).parent / "papers_with_metadata.csv"

ARXIV_API_URL = "http://export.arxiv.org/api/query?id_list={}"

def fetch_arxiv_metadata(arxiv_id):
    """Fetch metadata for a single arXiv paper."""
    url = ARXIV_API_URL.format(arxiv_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
        
        # Parse XML
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        
        title = entry.find('atom:title', ns)
        title = title.text.strip().replace('\n', ' ') if title is not None else ""
        
        # Get authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        
        # Get publication date
        published = entry.find('atom:published', ns)
        pub_date = published.text[:10] if published is not None else ""  # YYYY-MM-DD
        
        # Get categories
        categories = []
        for cat in entry.findall('atom:category', ns):
            term = cat.get('term')
            if term:
                categories.append(term)
        
        return {
            'title': title,
            'authors': '; '.join(authors[:5]) + ('...' if len(authors) > 5 else ''),
            'pub_date': pub_date,
            'categories': ', '.join(categories[:3])
        }
    except Exception as e:
        print(f"  Error fetching {arxiv_id}: {e}")
        return None

def main():
    # Read existing CSV
    papers = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
    
    print(f"Fetching metadata for {len(papers)} papers...")
    
    # Fetch metadata for each paper
    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        print(f"[{i+1}/{len(papers)}] Fetching {arxiv_id}...", end=' ')
        
        metadata = fetch_arxiv_metadata(arxiv_id)
        if metadata:
            paper['paper_title'] = metadata['title']
            paper['paper_authors'] = metadata['authors']
            paper['paper_pub_date'] = metadata['pub_date']
            print(f"OK - {metadata['title'][:50]}...")
        else:
            print("FAILED")
        
        # Rate limit: arXiv asks for 3 second delay
        time.sleep(0.5)  # Being nice to arXiv
    
    # Write output
    fieldnames = ['arxiv_id', 'found_in_artifacts', 'paper_title', 'paper_authors', 'paper_pub_date', 
                  'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'T1', 'T2', 'Q1', 'Q2', 'Q3']
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper)
    
    print(f"\nOutput written to: {OUTPUT_CSV}")
    
    # Summary
    with_title = sum(1 for p in papers if p.get('paper_title'))
    print(f"Successfully fetched: {with_title}/{len(papers)} papers")

if __name__ == "__main__":
    main()
