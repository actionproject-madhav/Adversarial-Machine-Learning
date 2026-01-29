#!/usr/bin/env python3
"""
Automated paper coding using GPT-4o.
Reads PDFs and codes them according to the 9-variable codebook.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()

import openai
from openai import OpenAI

# Try to import PDF reader
try:
    import fitz  # PyMuPDF - better than PyPDF2
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf")
    import fitz

# Configuration
BASE_DIR = Path(__file__).parent.parent
PAPERS_DIR = Path(__file__).parent / "papers"
CSV_FILE = Path(__file__).parent / "run1" / "papers_for_coding_71.csv"
OUTPUT_FILE = Path(__file__).parent / "run1" / "papers_coded_gpt4o.csv"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Coding prompt - detailed instructions for consistency
CODING_PROMPT = """You are coding academic papers for a systematic study of adversarial machine learning research.

For each paper, you must assign values to exactly 9 variables. Be consistent and precise.

## CODING VARIABLES

### G1: Paper Type
- **Attack**: Paper proposes a new attack method
- **Defense**: Paper proposes a defense/robustness method  
- **Evaluation**: Paper evaluates existing attacks/defenses without proposing new ones

### G2: Threat Category (for attacks only)
- **Evasion**: Fool model at test time (adversarial examples)
- **Poisoning**: Corrupt training data
- **Privacy**: Steal data or model (membership inference, model extraction, training data extraction)
- **N/A**: Use for defense papers

### G3: Domain
- **Vision**: Image classification (CIFAR, ImageNet, etc.)
- **NLP**: Text classification, sentiment, NER
- **LLM**: Large language models (GPT, Claude, jailbreaking, prompt injection)
- **Malware**: Malware detection
- **Audio**: Speech recognition, audio attacks
- **Tabular**: Tabular data, structured data
- **Cross-domain**: Paper evaluates on 2+ distinct domains

### G4: Publication Venue
- **ML**: NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV, ACL, EMNLP, NAACL
- **Security**: IEEE S&P, ACM CCS, USENIX Security, NDSS
- **Journal**: TPAMI, TIFS, TDSC, or other journals
- **arXiv-only**: No peer-reviewed venue (check if paper mentions acceptance)

### G5: Code Available
- **Yes**: Paper has public code repository (GitHub link in paper, or easily findable)
- **No**: No code available

### G6: Code Release Timing
- **At-pub**: Code released within 1 month of paper publication
- **Post-pub**: Code released later than 1 month after publication
- **Never**: No code released

### T1: Access Level (attacks only)
- **White**: Full model access (weights, gradients)
- **Gray**: Surrogate/substitute model used
- **Black**: Query access only, no model internals
- **N/A**: Use for defense papers

### T2: Gradient Required (attacks only)
- **Yes**: Attack uses gradients (backpropagation, gradient descent)
- **No**: Attack is gradient-free (e.g., evolutionary, random search)
- **N/A**: Use for defense papers

### Q1: Real-World Evaluation
- **Yes**: Tested on production system (Google API, Tesla, ChatGPT, commercial product)
- **Partial**: Realistic simulation or industry dataset
- **No**: Standard benchmarks only (CIFAR, ImageNet, MNIST)

## OUTPUT FORMAT

Return ONLY a JSON object with these exact keys:
{
    "G1": "Attack" or "Defense" or "Evaluation",
    "G2": "Evasion" or "Poisoning" or "Privacy" or "N/A",
    "G3": "Vision" or "NLP" or "LLM" or "Malware" or "Audio" or "Tabular" or "Cross-domain",
    "G4": "ML" or "Security" or "Journal" or "arXiv-only",
    "G5": "Yes" or "No",
    "G6": "At-pub" or "Post-pub" or "Never",
    "T1": "White" or "Gray" or "Black" or "N/A",
    "T2": "Yes" or "No" or "N/A",
    "Q1": "Yes" or "Partial" or "No",
    "reasoning": "Brief explanation of key decisions"
}

DO NOT include any text before or after the JSON object.
"""


def extract_pdf_text(pdf_path, max_pages=15):
    """Extract text from PDF, limiting to first N pages."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
        doc.close()
        
        # Limit text length to avoid token limits
        if len(text) > 50000:
            text = text[:50000] + "\n\n[TRUNCATED]"
        
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def find_pdf_for_paper(paper_title, papers_dir):
    """Find the PDF file matching a paper title."""
    # Clean the title for matching
    clean_title = re.sub(r'[^\w\s]', '', paper_title.lower())
    
    for pdf_file in papers_dir.glob("*.pdf"):
        clean_filename = re.sub(r'[^\w\s]', '', pdf_file.stem.lower())
        
        # Check if significant overlap
        title_words = set(clean_title.split())
        file_words = set(clean_filename.split())
        
        if len(title_words & file_words) >= min(3, len(title_words) - 1):
            return pdf_file
    
    return None


def code_paper(paper_text, paper_title, arxiv_id):
    """Call GPT-4o to code a paper."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CODING_PROMPT},
                {"role": "user", "content": f"Paper Title: {paper_title}\narXiv ID: {arxiv_id}\n\n---\n\n{paper_text}"}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            print(f"Could not parse JSON from response: {content[:200]}")
            return None
            
    except Exception as e:
        print(f"API error: {e}")
        return None


def main():
    print("=" * 60)
    print("AUTOMATED PAPER CODING WITH GPT-4o")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    # Load papers to code
    papers = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            papers.append(row)
    
    print(f"Loaded {len(papers)} papers from {CSV_FILE.name}")
    print(f"PDFs directory: {PAPERS_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Process each paper
    coded_papers = []
    errors = []
    
    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        title = paper['paper_title']
        
        print(f"[{i+1}/{len(papers)}] {arxiv_id}: {title[:50]}...", end=" ", flush=True)
        
        # Find PDF
        pdf_path = find_pdf_for_paper(title, PAPERS_DIR)
        
        if not pdf_path:
            print("PDF NOT FOUND")
            errors.append(arxiv_id)
            coded_papers.append(paper)
            continue
        
        # Extract text
        text = extract_pdf_text(pdf_path)
        if not text:
            print("TEXT EXTRACTION FAILED")
            errors.append(arxiv_id)
            coded_papers.append(paper)
            continue
        
        # Code paper
        result = code_paper(text, title, arxiv_id)
        
        if result:
            # Update paper with coding
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = result.get(col, '')
            print("OK")
        else:
            print("CODING FAILED")
            errors.append(arxiv_id)
        
        coded_papers.append(paper)
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(coded_papers)
    
    print()
    print("=" * 60)
    print(f"COMPLETE: {len(papers) - len(errors)}/{len(papers)} papers coded")
    print(f"Output saved to: {OUTPUT_FILE}")
    if errors:
        print(f"Errors ({len(errors)}): {', '.join(errors)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
