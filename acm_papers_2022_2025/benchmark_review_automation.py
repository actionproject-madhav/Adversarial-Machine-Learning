"""
ACM Adversarial ML Paper Analysis - OpenAI Version
Reads PDFs from ACM folder structure (2022-2025) and analyzes using GPT-4
Uses standardized benchmark criteria from benchmark_criteria.csv
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
from typing import Dict, List, Optional
import importlib.util

# Add parent directory to path to import benchmark_utils
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
benchmark_utils_path = parent_dir / "review_benchmarks_acm" / "benchmark_utils.py"

# Load benchmark_utils module dynamically
spec = importlib.util.spec_from_file_location("benchmark_utils", benchmark_utils_path)
benchmark_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_utils)

# Import functions from the loaded module
build_prompt_section = benchmark_utils.build_prompt_section
build_format_section = benchmark_utils.build_format_section
load_criteria = benchmark_utils.load_criteria

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def extract_text_from_pdf(pdf_path: str, max_pages: int = None) -> Dict[str, str]:
    """
    Extract text from PDF file
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to extract (None = all pages)
        
    Returns:
        dict with 'title', 'authors', 'abstract', 'full_text'
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            total_pages = len(pdf_reader.pages)
            
            # Extract pages (all if max_pages is None)
            text_parts = []
            pages_to_extract = total_pages if max_pages is None else min(max_pages, total_pages)
            
            for page_num in range(pages_to_extract):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            # Extract first page for metadata
            first_page = pdf_reader.pages[0].extract_text()
            lines = first_page.split('\n')
            
            # Try to extract title (usually in first 10 lines)
            title = ' '.join(lines[:10]).strip()[:200]  # First 200 chars
            
            # Try to extract authors (usually after title, before abstract)
            # Look for lines between title and abstract
            authors = ""
            for i, line in enumerate(lines[:30]):  # Search first 30 lines
                line_lower = line.lower().strip()
                # Common author indicators
                if any(keyword in line_lower for keyword in ['university', '@', 'author', 'department']):
                    # Take a few lines around this as authors
                    author_lines = lines[max(0, i-2):min(i+3, len(lines))]
                    authors = ' '.join(author_lines).strip()[:200]
                    break
            
            # If no authors found, try to get lines between 2-8 (typical author location)
            if not authors:
                authors = ' '.join(lines[2:8]).strip()[:200]
            
            # Try to find abstract
            abstract = ""
            if 'Abstract' in full_text or 'ABSTRACT' in full_text:
                # Simple extraction: text after "Abstract" keyword
                for variant in ['Abstract', 'ABSTRACT', 'Abstract—', 'ABSTRACT—']:
                    if variant in full_text:
                        start_idx = full_text.index(variant) + len(variant)
                        # Take next 1000 characters as abstract
                        abstract = full_text[start_idx:start_idx+1000].strip()
                        break
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract if abstract else full_text[:1000],
                'full_text': full_text  # ENTIRE paper text, no truncation
            }
            
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return {
            'title': Path(pdf_path).stem,
            'authors': '',
            'abstract': '',
            'full_text': ''
        }


def analyze_paper_openai(title: str, abstract: str, full_text: str = "") -> Dict:
    """
    Analyze paper using OpenAI GPT-4 with standardized benchmark criteria
    
    Args:
        title: Paper title
        abstract: Paper abstract
        full_text: Optional full text excerpt
        
    Returns:
        dict: Analysis results
    """
    
    # Build prompt using benchmark criteria from CSV
    prompt_questions = build_prompt_section()
    prompt_format = build_format_section()
    
    prompt = f"""You are analyzing an adversarial machine learning research paper. Answer ONLY with the exact options provided.

PAPER:
Title: {title}
Abstract: {abstract}
Full Text: {full_text}

{prompt_questions}

CRITICAL INSTRUCTIONS:
- Answer ONLY with the exact option words provided
- Do not add explanations
- Format EXACTLY as shown below
- If unclear from text, make best judgment

{prompt_format}"""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4o-mini" for lower cost
            messages=[
                {"role": "system", "content": "You are a precise research paper analyzer. Answer only with exact specified options."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Deterministic responses
            max_tokens=300
        )
        
        # Extract response
        response_text = response.choices[0].message.content
        
        # Parse response
        results = parse_response(response_text)
        
        return results
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def parse_response(response_text: str) -> Dict:
    """Parse API response into structured data"""
    results = {}
    
    lines = response_text.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()
                # Remove brackets if present
                answer = answer.strip('[]').strip()
                results[question] = answer
    
    return results


def calculate_flags(results: Dict) -> Dict:
    """Calculate binary flags and gap score"""
    flags = {}
    
    # Traditional academic flags (1 = problematic for practice)
    flags['Flag_Grad'] = 1 if results.get('Q1') == 'YES' else 0
    flags['Flag_HighQ'] = 1 if results.get('Q2') == 'High' else 0
    flags['Flag_WB'] = 1 if results.get('T1') == 'White-box' else 0
    flags['Flag_NoEcon'] = 1 if results.get('G5') == 'NO' else 0
    flags['Flag_NoCode'] = 1 if results.get('G6') == 'NO' else 0
    flags['Flag_NoReal'] = 1 if results.get('G7') == 'NO' else 0
    
    # Total gap score
    flags['Traditional_Score'] = sum([
        flags['Flag_Grad'],
        flags['Flag_HighQ'],
        flags['Flag_WB'],
        flags['Flag_NoEcon'],
        flags['Flag_NoCode'],
        flags['Flag_NoReal']
    ])
    
    return flags


def scan_pdf_folder(base_path: str) -> List[Dict]:
    """
    Scan folder structure for PDFs and organize by year
    
    Args:
        base_path: Path to acm_papers_2022_2025 folder (or current directory)
        
    Returns:
        List of dicts with paper info
    """
    papers = []
    
    base_path = Path(base_path).resolve()  # Resolve to absolute path
    
    # Debug: Show what path we're scanning
    if not base_path.exists():
        print(f"ERROR: Base path does not exist: {base_path}")
        return papers
    
    # Scan each year folder (2022, 2023, 2024, 2025)
    for year_folder in sorted(base_path.iterdir()):
        if year_folder.is_dir() and year_folder.name.isdigit():
            year = year_folder.name
            
            # Only process years 2022-2025
            if not (2022 <= int(year) <= 2025):
                continue
            
            # Get all PDFs in this year
            pdf_files = list(year_folder.glob("*.pdf"))
            
            print(f"Found {len(pdf_files)} PDFs in {year}/")
            
            for pdf_file in pdf_files:
                papers.append({
                    'year': year,
                    'filename': pdf_file.name,
                    'filepath': str(pdf_file)
                })
    
    return papers


def analyze_all_papers(base_path: str, output_csv: str, delay: float = 1.0, resume: bool = True):
    """
    Analyze all PDFs in folder structure
    
    Args:
        base_path: Path to acm_papers_2022_2025 folder (or current directory)
        output_csv: Path to save results CSV
        delay: Delay between API calls (seconds)
        resume: If True, skip already analyzed papers
    """
    
    print("="*80)
    print("ACM ADVERSARIAL ML PAPER ANALYSIS (2022-2025)")
    print("="*80)
    
    # Load and display benchmark criteria being used
    print("\nUsing standardized benchmark criteria from benchmark_criteria.csv")
    criteria = load_criteria()
    print(f"  Total questions: {len(criteria)}")
    print(f"  Categories: {', '.join(criteria['Category'].unique())}")
    
    # Scan for papers
    print("\nScanning for PDFs...")
    print(f"Base path: {Path(base_path).resolve()}")
    papers = scan_pdf_folder(base_path)
    
    if len(papers) == 0:
        print(f"\nNo PDFs found! Check the path: {Path(base_path).resolve()}")
        print(f"Path exists: {Path(base_path).resolve().exists()}")
        print(f"Contents: {list(Path(base_path).resolve().iterdir())}")
        return
    
    # Check for existing results (resume capability)
    already_analyzed = set()
    if resume and Path(output_csv).exists():
        print(f"\n✓ Found existing results: {output_csv}")
        df_existing = pd.read_csv(output_csv)
        already_analyzed = set(df_existing['Filename'].values)
        print(f"  Already analyzed: {len(already_analyzed)} papers")
        
        # Filter out already analyzed papers
        papers = [p for p in papers if p['filename'] not in already_analyzed]
        print(f"  Remaining to analyze: {len(papers)} papers")
    
    total_papers = len(papers)
    print(f"\nTotal papers to process: {total_papers}")
    
    if total_papers == 0:
        print("All papers have been analyzed already!")
        # Still print statistics from existing file
        if Path(output_csv).exists():
            df_existing = pd.read_csv(output_csv)
            print_statistics(df_existing)
        return
    
    # Show breakdown by year
    print("\n--- BREAKDOWN BY YEAR ---")
    from collections import Counter
    year_counts = Counter(p['year'] for p in papers)
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]} papers")
    
    # Confirm before proceeding
    estimated_cost = total_papers * 0.15  # ~$0.15 per paper with full text
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"Estimated time: {total_papers * (delay + 2):.0f} seconds (~{total_papers * (delay + 2) / 60:.1f} minutes)")
    
    response = input("\nProceed with analysis? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Process each paper
    results_list = []
    
    for idx, paper in enumerate(papers):
        print(f"\n[{idx+1}/{total_papers}] Processing: {paper['filename']} ({paper['year']})")
        
        # Extract text from PDF
        print("  - Extracting text from PDF...")
        extracted = extract_text_from_pdf(paper['filepath'], max_pages=None)  # Extract ALL pages
        full_text_length = len(extracted['full_text'])
        estimated_tokens = full_text_length // 4
        print(f"  - Extracted {full_text_length:,} characters (~{estimated_tokens:,} tokens) - Full paper included")
        
        # Analyze with OpenAI
        print("  - Analyzing with GPT-4...")
        results = analyze_paper_openai(
            title=extracted['title'],
            abstract=extracted['abstract'],
            full_text=extracted['full_text']
        )
        
        if results:
            # Calculate flags
            flags = calculate_flags(results)
            
            # Combine all data
            paper_data = {
                'Year': paper['year'],
                'Filename': paper['filename'],
                'Title': extracted['title'][:150],  # Truncate for CSV
                'Authors': extracted['authors'][:200],  # Add authors
                **results,
                **flags
            }
            
            results_list.append(paper_data)
            print(f"  ✓ Traditional Score: {flags['Traditional_Score']}/6")
        else:
            print(f"  ✗ Analysis failed")
        
        # Rate limiting
        time.sleep(delay)
    
    # Create DataFrame
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(results_list)
    
    # Append to existing CSV if resume mode
    if resume and Path(output_csv).exists():
        print(f"\n✓ Appending {len(df_results)} new results to existing file")
        df_existing = pd.read_csv(output_csv)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    
    # Sort by Year (2022-2025) and then by Filename for consistent ordering
    df_results = df_results.sort_values(['Year', 'Filename'], ascending=[True, True]).reset_index(drop=True)
    
    # Save to CSV
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    print(f"  Total papers in CSV: {len(df_results)}")
    
    # Print statistics
    print_statistics(df_results)
    
    return df_results


def print_statistics(df: pd.DataFrame):
    """Print comprehensive summary statistics based on benchmark criteria"""
    
    print("\n" + "="*80)
    print("ACM PAPERS SUMMARY STATISTICS (2022-2025)")
    print("="*80)
    
    total = len(df)
    print(f"\nTotal papers analyzed: {total}")
    
    # By year
    print("\n--- BY YEAR ---")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        pct = count / total * 100
        print(f"  {year}: {count} papers ({pct:.1f}%)")
    
    print("\n--- BASIC CHARACTERISTICS ---")
    print(f"% Attack papers: {(df['G1'] == 'atk').sum() / total * 100:.1f}%")
    print(f"% Defense papers: {(df['G1'] == 'def').sum() / total * 100:.1f}%")
    print(f"% Both attack & defense: {(df['G1'] == 'both').sum() / total * 100:.1f}%")
    print(f"% Deep learning only: {(df['G3'] == 'DL').sum() / total * 100:.1f}%")
    print(f"% Traditional ML only: {(df['G3'] == 'Traditional').sum() / total * 100:.1f}%")
    print(f"% Image data: {(df['G4'] == 'Images').sum() / total * 100:.1f}%")
    print(f"% Text data: {(df['G4'] == 'Text').sum() / total * 100:.1f}%")
    
    print("\n--- PRACTICAL INDICATORS (Higher = Better) ---")
    econ_pct = (df['G5'] == 'YES').sum() / total * 100
    code_pct = (df['G6'] == 'YES').sum() / total * 100
    real_pct = (df['G7'] == 'YES').sum() / total * 100
    print(f"% Mention economics/cost: {econ_pct:.1f}%")
    print(f"% Release code: {code_pct:.1f}%")
    print(f"% Test real systems: {real_pct:.1f}%")
    
    print("\n--- ACADEMIC INDICATORS / GAP FACTORS (Lower = Better) ---")
    grad_pct = df['Flag_Grad'].sum() / total * 100
    highq_pct = df['Flag_HighQ'].sum() / total * 100
    wb_pct = df['Flag_WB'].sum() / total * 100
    print(f"% Gradient-based methods: {grad_pct:.1f}%")
    print(f"% High query budget required: {highq_pct:.1f}%")
    print(f"% White-box only: {wb_pct:.1f}%")
    print(f"% No economics mentioned: {df['Flag_NoEcon'].sum() / total * 100:.1f}%")
    print(f"% No code released: {df['Flag_NoCode'].sum() / total * 100:.1f}%")
    print(f"% No real system testing: {df['Flag_NoReal'].sum() / total * 100:.1f}%")
    
    print("\n--- GAP ANALYSIS ---")
    avg_score = df['Traditional_Score'].mean()
    print(f"Average Gap Score: {avg_score:.2f}/6 (lower = more practical)")
    
    high_gap = (df['Traditional_Score'] >= 4).sum()
    med_gap = ((df['Traditional_Score'] >= 2) & (df['Traditional_Score'] < 4)).sum()
    low_gap = (df['Traditional_Score'] < 2).sum()
    
    print(f"\nHigh Gap (4-6): {high_gap} papers ({high_gap/total*100:.1f}%)")
    print(f"Medium Gap (2-3): {med_gap} papers ({med_gap/total*100:.1f}%)")
    print(f"Low Gap (0-1): {low_gap} papers ({low_gap/total*100:.1f}%)")
    
    # Gap score distribution
    print("\n--- GAP SCORE DISTRIBUTION ---")
    score_dist = df['Traditional_Score'].value_counts().sort_index()
    for score in range(7):
        count = score_dist.get(score, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)  # Visual bar
        print(f"  Score {score}: {count:3d} papers ({pct:5.1f}%) {bar}")
    
    # Threat model analysis
    print("\n--- THREAT MODEL ANALYSIS ---")
    print(f"% White-box attacks: {(df['T1'] == 'White-box').sum() / total * 100:.1f}%")
    print(f"% Gray-box attacks: {(df['T1'] == 'Gray-box').sum() / total * 100:.1f}%")
    print(f"% Black-box attacks: {(df['T1'] == 'Black-box').sum() / total * 100:.1f}%")
    
    # Attack types
    print("\n--- ATTACK TYPE BREAKDOWN ---")
    attack_types = df['G2'].value_counts()
    for atk_type, count in attack_types.items():
        print(f"  {atk_type}: {count} papers ({count/total*100:.1f}%)")
    
    print("\n" + "="*80)


# Main execution
if __name__ == "__main__":
    """
    USAGE:
    
    1. Set OPENAI_API_KEY in .env file (in parent directory):
       OPENAI_API_KEY=sk-...
       
    2. Run from the acm_papers_2022_2025/ directory:
       python benchmark_review_automation.py
       
    OR from parent directory:
       cd acm_papers_2022_2025 && python benchmark_review_automation.py
       
    The script will:
    - Scan all year folders (2022, 2023, 2024, 2025)
    - Extract text from all PDFs
    - Analyze each paper with GPT-4 using benchmark criteria
    - Save results to CSV
    - Print comprehensive statistics
    """
    
    # Configuration
    # BASE_PATH should be the script's directory (acm_papers_2022_2025)
    script_dir = Path(__file__).parent.resolve()
    BASE_PATH = str(script_dir)  # Use script's directory as base path
    OUTPUT_CSV = str(script_dir / "acm_analysis_results_2022_2025.csv")
    DELAY = 1.0  # seconds between API calls
    
    # Check if year folders exist
    year_folders = [d for d in script_dir.iterdir() if d.is_dir() and d.name.isdigit() and 2022 <= int(d.name) <= 2025]
    
    if not year_folders:
        print(f"Error: No year folders (2022-2025) found in {script_dir}")
        print("\nCurrent directory:", os.getcwd())
        print(f"\nPlease run from the acm_papers_2022_2025/ directory")
        print("Expected structure:")
        print("  acm_papers_2022_2025/")
        print("    ├── 2022/")
        print("    ├── 2023/")
        print("    ├── 2024/")
        print("    ├── 2025/")
        print("    └── benchmark_review_automation.py")
    else:
        print(f"Found {len(year_folders)} year folders: {[d.name for d in year_folders]}")
        # Run analysis
        analyze_all_papers(BASE_PATH, OUTPUT_CSV, delay=DELAY)

        