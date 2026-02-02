# Research Project Summary: Adversarial ML Adoption Lag Study

## Research Topic

**Tracing the Adoption of Adversarial Machine Learning Research into Industry Practice (2014-2025)**

Measuring the temporal lag between publication of adversarial ML research and demonstrable industry adoption through tools, benchmarks, and regulatory frameworks.

---

## Core Research Questions

### RQ1: Adoption Lag

What is the typical time lag between publication of landmark AML research and evidence of industry adoption, measured through tool integration, commercial reference, regulatory citation, and production deployment?

### RQ2: Domain Variation

How does adoption speed vary across application domains (computer vision, NLP, malware detection, autonomous systems, LLMs), and what factors explain these differences?

### RQ3: Acceleration Factors

What mechanisms—regulatory frameworks, standardized benchmarks, industry consortiums, commercial tools—have accelerated adoption, particularly for foundation model security post-2022?

---

## Methodology Overview

### Artifact-Anchored Backward Traceability

**Key Innovation**: Instead of starting from papers and speculating about impact, we start from industry artifacts and trace backwards to source papers. This ensures every paper in our sample has verified adoption evidence.

### Data Sources (9 Git-Searchable Artifacts)

#### Open-Source Tools (≥1,000 GitHub stars)

1. **CleverHans** (6,401 stars) - First major AML library
2. **IBM ART** (5,789 stars) - Adversarial Robustness Toolbox
3. **TextAttack** (3,348 stars) - NLP adversarial attacks
4. **PyRIT** (3,343 stars) - LLM red-teaming (Microsoft)
5. **Foolbox** (2,936 stars) - Adversarial attacks (Bethge Lab)

#### Standardized Benchmarks (Peer-reviewed)

6. **RobustBench** (NeurIPS 2021) - Adversarial robustness leaderboard
7. **AutoAttack** (ICML 2020) - Standardized attack ensemble
8. **HarmBench** (ICML 2024) - LLM jailbreak evaluation

#### Regulatory Frameworks

9. **MITRE ATLAS** - Industry-standard adversarial ML threat framework (66 techniques, 33 case studies)

### Paper Selection Process

1. **Automated Extraction**: Extract all arXiv references from the 9 artifacts via Git repository scanning
2. **Initial Pool**: 277 unique AML papers extracted
3. **Selection Criteria**:
   - Multi-artifact papers (≥2 artifacts): 61 papers
   - MITRE ATLAS-only papers: 10 papers
4. **Final Sample**: **71 papers** for manual coding

### Adoption Event Definitions

- **Tool adoption**: First Git commit referencing the paper
- **Benchmark adoption**: Paper cited in benchmark repository
- **Regulatory adoption**: Paper referenced in MITRE ATLAS

### Adoption Lag Calculation

```
Adoption Lag = Date_artifact - Date_publication
```

- **Artifact dates**: First Git commit timestamp (UTC)
- **Publication dates**: Conference/journal date OR first arXiv submission
- **First adoption**: Earliest adoption across all artifacts

---

## Coding Framework (12 Variables)

### Research Characteristics (G1-G6)

| Variable               | Values                                                        | Description               |
| ---------------------- | ------------------------------------------------------------- | ------------------------- |
| **G1: Type**           | Attack / Defense / Evaluation                                 | Primary contribution      |
| **G2: Threat**         | Evasion / Poisoning / Privacy / N/A                           | Attack category           |
| **G3: Domain**         | Vision / NLP / LLM / Malware / Audio / Tabular / Cross-domain | Primary evaluation domain |
| **G4: Venue**          | ML / Security / Journal / arXiv-only                          | Publication venue type    |
| **G5: Code available** | Yes / No                                                      | Code link exists          |
| **G6: Code timing**    | At-pub / Post-pub / Never                                     | When code was released    |

### Threat Model (T1-T2, Attack papers only)

| Variable                  | Values               | Description                 |
| ------------------------- | -------------------- | --------------------------- |
| **T1: Access level**      | White / Gray / Black | Model access assumptions    |
| **T2: Gradient required** | Yes / No             | Gradients used at any stage |

### Practical Evaluation (Q1)

| Variable                | Values             | Description              |
| ----------------------- | ------------------ | ------------------------ |
| **Q1: Real-world eval** | Yes / Partial / No | Production system tested |

---

## Statistical Analysis Plan

### RQ1: Adoption Lag Analysis

- **Descriptive**: Median, IQR, range of adoption lags
- **Stratification**: By artifact type, publication era (2014-2017, 2018-2021, 2022-2025)
- **Tests**: Kruskal-Wallis with post-hoc Dunn tests

### RQ2: Domain Variation

- **Comparison**: Adoption lags across 7 domains
- **Tests**: Pairwise Mann-Whitney U with Bonferroni correction (α = 0.05/21 = 0.0024)
- **Hypothesis**: LLM papers show significantly shorter lags than CV papers

### RQ3: Acceleration Factors

- **Model**: Cox proportional hazards regression
- **Outcome**: Time-to-first-adoption
- **Covariates**:
  - Publication year (continuous)
  - Domain (categorical, ref: Vision)
  - Venue type (ML vs. Security, ref: ML)
  - Code availability (binary)
  - Threat model (white/gray/black, ref: white-box)
- **Interpretation**: Hazard ratios >1 indicate faster adoption
- **Diagnostics**: Schoenfeld residuals for proportional hazards assumption

---

## Data Files

### Input Data

- `extraction_runs/run1/papers_for_coding_71.csv` - Papers to code (71 papers)
- `extraction_runs/run1/papers_coded_verified_fixed.csv` - Coded papers with adoption dates

### Coding Instructions

- `coding_instructions.pdf` - Detailed codebook and decision rules

### Analysis Scripts

- `fix_adoption_dates.py` - Script to extract adoption dates from Git history

---

## Key Contributions

1. **Artifact-anchored methodology**: Reverse-engineering approach ensuring verified adoption evidence
2. **Reproducible coding framework**: 12-variable codebook with explicit decision rules
3. **Quantified adoption analysis**: First systematic measurement of research-to-industry lag across domains and eras

---

## Expected Outputs

### Tables

1. Sample construction by artifact source
2. Descriptive statistics for coded papers
3. Adoption lag by artifact type
4. Adoption lag by publication era
5. Adoption lag by domain
6. Cox regression results (hazard ratios)

### Figures

1. Adoption lag distributions (box plots/violin plots)
2. Domain comparison visualization
3. Forest plot of hazard ratios

### Deliverables

- Structured CSV dataset with all coding decisions
- Codebook documentation
- Statistical analysis code
- Complete research paper

---

## Timeline Context

- **Foundational era (2014-2017)**: Szegedy, Goodfellow FGSM, C&W, Madry PGD
- **Expansion era (2018-2021)**: Privacy attacks, backdoors, physical attacks
- **LLM era (2022-2025)**: Prompt injection, jailbreaking, alignment techniques

---

## Research Gap Being Addressed

Prior work characterized the research-practice gap **qualitatively** (Apruzzese 2023, Kumar 2020, Grosse 2023, Mink 2023). This study provides the first **quantitative measurement** of adoption timelines, identifying specific lag durations and acceleration factors.

Submit to ACM CCS (April 29)and if rejected submit to USENIX Security (August)
