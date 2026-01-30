# Tracing the Adoption of Adversarial Machine Learning Research into Industry Practice

Systematic review measuring the time lag between publication of adversarial ML (AML) research and evidence of industry adoption. The study covers 2014–2025 and uses an artifact-anchored methodology: it starts from concrete industry artifacts (tools, benchmarks, regulatory frameworks) and traces each adopted technique back to the originating academic paper.

## Research questions

- **RQ1 (Adoption lag):** Typical time between paper publication and industry adoption, measured via tool integration, benchmark inclusion, regulatory citation.
- **RQ2 (Domain variation):** How adoption speed differs across domains (vision, NLP, malware, LLMs) and what explains it.
- **RQ3 (Acceleration factors):** Which mechanisms (regulatory frameworks, benchmarks, commercial tools) have sped adoption, especially for foundation-model security after 2022.

## Artifacts

**Tools** (included if they have at least 1,000 GitHub stars): CleverHans, IBM Adversarial Robustness Toolbox, Foolbox, TextAttack, PyRIT.

**Benchmarks** (peer-reviewed): RobustBench, AutoAttack, HarmBench.

**Regulatory:** MITRE ATLAS (adversarial ML threat framework).

## Repository layout

- **`artifact_repos/`** — Cloned or mirrored copies of the above tools and benchmarks. The extraction pipeline scans these for arXiv references.
- **`extraction_runs/`** — Extraction scripts and per-run outputs:
  - **`run_extraction.py`** — Scans artifact repos for arXiv IDs (via URLs, BibTeX, inline citations), resolves them with the arXiv API, and writes paper lists per artifact and combined. Run with `run1`, `run2`, or `compare`.
  - **`auto_code_papers.py`** — Automated coding of papers using a 9-variable codebook: reads PDFs from `extraction_runs/papers/`, calls GPT-4o, writes coded CSV.
  - **`run1/`** — Example run: `papers_all.csv`, `papers_by_artifact.csv`, `papers_for_coding_71.csv`, `papers_coded_verified.csv`, and related files.
- **`coding_instructions.pdf`** / **`coding_instructions.tex`** — Human coding codebook and decision rules.
- **`main.tex`** — Paper describing the study design, methodology, and results.

## Coding scheme (quick reference)

| Code | Meaning | Values |
|------|---------|--------|
| G1 | Goal | Attack, Defense, Evaluation, or combinations |
| G2 | Attack type | Evasion, Poisoning, Defense, N/A |
| G3 | Domain | Vision, NLP, LLM, Audio, Malware, Cross-domain |
| G4 | Venue | ML, Security, arXiv-only |
| G5 | Code released | Yes, No |
| G6 | Code timing | At-pub, Never |
| T1 | Threat model | White, Black, White/Black, N/A |
| T2 | Targeted attack | Yes, No, N/A |
| Q1 | Query-based / real-world eval | Yes, No, Partial, N/A |

## Requirements

- Python 3 with: `openai`, `python-dotenv`, `pymupdf` (for `auto_code_papers.py`).
- `OPENAI_API_KEY` in `.env` for automated coding.
- For extraction: network access for the arXiv API; artifact repos under `artifact_repos/` (clone or copy as needed).

## Usage

1. **Extract papers from artifacts:**  
   `python extraction_runs/run_extraction.py run1`  
   Outputs go under `extraction_runs/run1/`.

2. **Code papers (with PDFs in `extraction_runs/papers/`):**  
   `python extraction_runs/auto_code_papers.py`  
   Reads the run’s `papers_for_coding_*.csv` and writes coded CSV in the same run folder.

The paper (`main.tex`) defines the full methodology, artifact selection criteria, adoption event definitions, and exclusion rules.
