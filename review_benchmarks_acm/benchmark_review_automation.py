"""
Adversarial ML Paper Analysis - OpenAI Version
Reads PDFs from folder structure and analyzes using GPT-4
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> Dict[str, str]:
    """
    Extract text from PDF file
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to extract (for cost control)
        
    Returns:
        dict with 'title', 'authors', 'abstract', 'full_text'
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            total_pages = len(pdf_reader.pages)
            
            # Extract first few pages (usually contains title, abstract, intro)
            text_parts = []
            pages_to_extract = min(max_pages, total_pages)
            
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
                'full_text': full_text[:4000]  # First 4000 chars for context
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
    Analyze paper using OpenAI GPT-4
    
    Args:
        title: Paper title
        abstract: Paper abstract
        full_text: Optional full text excerpt
        
    Returns:
        dict: Analysis results
    """
    
    # Construct the prompt
    prompt = f"""You are analyzing an adversarial machine learning research paper. Answer ONLY with the exact options provided.

PAPER:
Title: {title}
Abstract: {abstract}
Text excerpt: {full_text[:2000]}

ANSWER THESE 12 QUESTIONS WITH ONLY THE SPECIFIED OPTIONS:

G1. Focus - Main contribution?
OPTIONS: atk, def, both
ANSWER: [atk for attack paper, def for defense paper, both for combined]

G2. Attack Type - What kind of attack/threat?
OPTIONS: Evasion, Poisoning, Privacy, Multiple, NA
ANSWER: [Evasion=test-time, Poisoning=training-time, Privacy=membership/extraction, Multiple=several types, NA=defense only]

G3. ML Type - Machine learning approach?
OPTIONS: DL, Traditional, Both
ANSWER: [DL=deep learning only, Traditional=classical ML, Both=both types]

G4. Data Type - What data is evaluated?
OPTIONS: Images, Text, Audio, Malware, Other
ANSWER: [Pick one primary type]

G5. Economics - Cost/resources/economics mentioned?
OPTIONS: YES, NO
ANSWER: [YES if ANY mention of cost/resources/economics, NO otherwise]

G6. Code Released - Source code publicly available?
OPTIONS: YES, NO
ANSWER: [YES if code is available, NO if not mentioned or not available]

G7. Real System - Tested on real/commercial systems?
OPTIONS: YES, NO
ANSWER: [YES if tested on deployed/commercial/production system, NO if only academic datasets/models]

T1. Model Knowledge - What does attacker know?
OPTIONS: White-box, Gray-box, Black-box
ANSWER: [White-box=full access to model, Gray-box=partial knowledge/surrogate, Black-box=no model knowledge]

T2. Training Data - Attacker's training data access?
OPTIONS: Full, Partial, None
ANSWER: [Full=complete access, Partial=subset/surrogate data, None=no training data]

Q1. Gradients - Requires computing gradients?
OPTIONS: YES, NO
ANSWER: [YES if method requires gradient computation, NO if gradient-free]

Q2. Query Budget - How many queries needed?
OPTIONS: High, Low, None
ANSWER: [High=>1000 queries, Low=<1000 queries, None=no queries needed]

Q3. Computation - Computational resources needed?
OPTIONS: High, Low
ANSWER: [High=needs GPU/significant time, Low=CPU ok/quick execution]

CRITICAL INSTRUCTIONS:
- Answer ONLY with the exact option words provided
- Do not add explanations
- Format EXACTLY as shown below
- If unclear from text, make best judgment

REQUIRED FORMAT:
G1: [answer]
G2: [answer]
G3: [answer]
G4: [answer]
G5: [answer]
G6: [answer]
G7: [answer]
T1: [answer]
T2: [answer]
Q1: [answer]
Q2: [answer]
Q3: [answer]"""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4o-mini" for lower cost
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
        base_path: Path to acm_papers_2022_2025 folder
        
    Returns:
        List of dicts with paper info
    """
    papers = []
    
    base_path = Path(base_path)
    
    # Scan each year folder
    for year_folder in sorted(base_path.iterdir()):
        if year_folder.is_dir() and year_folder.name.isdigit():
            year = year_folder.name
            
            # Get all PDFs in this year
            pdf_files = list(year_folder.glob("*.pdf"))
            
            print(f"\nFound {len(pdf_files)} PDFs in {year}/")
            
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
        base_path: Path to acm_papers_2022_2025 folder
        output_csv: Path to save results CSV
        delay: Delay between API calls (seconds)
        resume: If True, skip already analyzed papers
    """
    
    print("="*80)
    print("ADVERSARIAL ML PAPER ANALYSIS")
    print("="*80)
    
    # Scan for papers
    print("\nScanning for PDFs...")
    papers = scan_pdf_folder(base_path)
    
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
        print("No PDFs found! Check the path.")
        return
    
    # Confirm before proceeding
    estimated_cost = total_papers * 0.05  # ~$0.05 per paper with GPT-4
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"Estimated time: {total_papers * (delay + 2):.0f} seconds")
    
    response = input("\nProceed with analysis? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Process each paper
    results_list = []
    
    for idx, paper in enumerate(papers):
        print(f"\n[{idx+1}/{total_papers}] Processing: {paper['filename']}")
        
        # Extract text from PDF
        print("  - Extracting text from PDF...")
        extracted = extract_text_from_pdf(paper['filepath'], max_pages=5)
        
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
    
    # Save to CSV
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    print(f"  Total papers in CSV: {len(df_results)}")
    
    # Print statistics
    print_statistics(df_results)
    
    return df_results


def print_statistics(df: pd.DataFrame):
    """Print summary statistics"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total = len(df)
    print(f"\nTotal papers analyzed: {total}")
    
    # By year
    print("\n--- BY YEAR ---")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"{year}: {count} papers")
    
    print("\n--- BASIC CHARACTERISTICS ---")
    print(f"% Attack papers: {(df['G1'] == 'atk').sum() / total * 100:.1f}%")
    print(f"% Defense papers: {(df['G1'] == 'def').sum() / total * 100:.1f}%")
    print(f"% Deep learning only: {(df['G3'] == 'DL').sum() / total * 100:.1f}%")
    print(f"% Image data: {(df['G4'] == 'Images').sum() / total * 100:.1f}%")
    
    print("\n--- PRACTICAL INDICATORS (Good) ---")
    print(f"% Mention economics: {(df['G5'] == 'YES').sum() / total * 100:.1f}%")
    print(f"% Release code: {(df['G6'] == 'YES').sum() / total * 100:.1f}%")
    print(f"% Test real systems: {(df['G7'] == 'YES').sum() / total * 100:.1f}%")
    
    print("\n--- ACADEMIC INDICATORS (Gap) ---")
    print(f"% Gradient-based: {df['Flag_Grad'].sum() / total * 100:.1f}%")
    print(f"% High queries: {df['Flag_HighQ'].sum() / total * 100:.1f}%")
    print(f"% White-box: {df['Flag_WB'].sum() / total * 100:.1f}%")
    
    print("\n--- GAP ANALYSIS ---")
    avg_score = df['Traditional_Score'].mean()
    print(f"Average Gap Score: {avg_score:.2f}/6")
    
    high_gap = (df['Traditional_Score'] >= 4).sum()
    med_gap = ((df['Traditional_Score'] >= 2) & (df['Traditional_Score'] < 4)).sum()
    low_gap = (df['Traditional_Score'] < 2).sum()
    
    print(f"\nHigh Gap (4-6): {high_gap} papers ({high_gap/total*100:.1f}%)")
    print(f"Medium Gap (2-3): {med_gap} papers ({med_gap/total*100:.1f}%)")
    print(f"Low Gap (0-1): {low_gap} papers ({low_gap/total*100:.1f}%)")
    
    # Comparison with Apruzzese
    print("\n" + "="*80)
    print("COMPARISON WITH APRUZZESE 2019-2021")
    print("="*80)
    
    comparisons = [
        ("Attack focus", 72, (df['G1'] == 'atk').sum() / total * 100),
        ("DL only", 89, (df['G3'] == 'DL').sum() / total * 100),
        ("No economics", 73, df['Flag_NoEcon'].sum() / total * 100),
        ("Code released", 51, (df['G6'] == 'YES').sum() / total * 100),
        ("Real systems", 20, (df['G7'] == 'YES').sum() / total * 100),
    ]
    
    print(f"\n{'Metric':<20} {'2019-21':<10} {'2022-25':<10} {'Change':<10}")
    print("-"*60)
    for metric, old, new in comparisons:
        change = new - old
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"{metric:<20} {old:>6.1f}%   {new:>6.1f}%   {direction} {abs(change):>5.1f}%")


# Main execution
if __name__ == "__main__":
    """
    USAGE:
    
    1. Set OPENAI_API_KEY in .env file:
       OPENAI_API_KEY=sk-...
       
    2. Run:
       python analyze_papers_openai.py
       
    The script will:
    - Scan acm_papers_2022_2025/ folder
    - Extract text from all PDFs
    - Analyze each paper with GPT-4
    - Save results to CSV
    - Print statistics
    """
    
    # Configuration
    BASE_PATH = "acm_papers_2022_2025"
    OUTPUT_CSV = "analysis_results.csv"
    DELAY = 1.0  # seconds between API calls
    
    # Check if path exists
    if not Path(BASE_PATH).exists():
        print(f"Error: Path '{BASE_PATH}' not found!")
        print("\nCurrent directory:", os.getcwd())
        print("\nPlease run from the Adversarial-Machine-Learning/ directory")
        print("Or update BASE_PATH in the script")
    else:
        # Run analysis
        analyze_all_papers(BASE_PATH, OUTPUT_CSV, delay=DELAY)