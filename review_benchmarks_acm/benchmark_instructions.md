# Adversarial ML Paper Benchmark Criteria

This document describes the standardized benchmark criteria used for evaluating adversarial machine learning research papers.

## Overview

The benchmark consists of **12 questions** organized into 3 categories:

- **General (G1-G7)**: 7 questions about paper characteristics
- **Threat Model (T1-T2)**: 2 questions about attack assumptions
- **Query/Computation (Q1-Q3)**: 3 questions about practical requirements

## Reference

The complete criteria specification is available in `benchmark_criteria.csv`, which provides:
- Question IDs and categories
- Full question text
- Valid options for each question
- Answer guidelines
- Format examples

## Usage

### Loading the Criteria

```python
import pandas as pd

# Load the benchmark criteria
criteria = pd.read_csv('benchmark_criteria.csv')

# Access specific question
g1 = criteria[criteria['Question_ID'] == 'G1'].iloc[0]
print(f"{g1['Question_Title']}: {g1['Options']}")
```

### Building Prompts

The CSV can be used to dynamically generate analysis prompts:

```python
def build_question_prompt(criteria_df):
    prompt_parts = []
    for _, row in criteria_df.iterrows():
        prompt_parts.append(
            f"{row['Question_ID']}. {row['Question_Title']} - {row['Question_Description']}\n"
            f"OPTIONS: {row['Options']}\n"
            f"ANSWER: {row['Answer_Guidelines']}"
        )
    return "\n\n".join(prompt_parts)
```

## Critical Instructions

When using these criteria:

1. **Answer ONLY with the exact option words provided**
2. **Do not add explanations**
3. **Format EXACTLY as shown in Format_Example column**
4. **If unclear from text, make best judgment**

## Required Format

Answers should be formatted as:
```
G1: [answer]
G2: [answer]
G3: [answer]
...
Q3: [answer]
```

## Gap Score Calculation

The benchmark is used to calculate a "gap score" indicating how far research is from practical deployment:

- **Flag_Grad**: Requires gradients (Q1 == 'YES')
- **Flag_HighQ**: High query budget (Q2 == 'High')
- **Flag_WB**: White-box only (T1 == 'White-box')
- **Flag_NoEcon**: No economics mentioned (G5 == 'NO')
- **Flag_NoCode**: No code released (G6 == 'NO')
- **Flag_NoReal**: No real system testing (G7 == 'NO')

**Traditional_Score** = Sum of all flags (0-6, higher = larger gap)

## Categories

### General Questions (G1-G7)
- **G1**: Paper focus (attack/defense/both)
- **G2**: Type of attack/threat
- **G3**: ML approach type
- **G4**: Primary data type
- **G5**: Economics/cost considerations
- **G6**: Code availability
- **G7**: Real system testing

### Threat Model (T1-T2)
- **T1**: Attacker's model knowledge
- **T2**: Attacker's training data access

### Query/Computation (Q1-Q3)
- **Q1**: Gradient requirements
- **Q2**: Query budget needs
- **Q3**: Computational resources

## File Structure

- `benchmark_criteria.csv`: Machine-readable specification of all criteria
- `benchmark_instructions.md`: This file - human-readable documentation
- `benchmark_review_automation.py`: Implementation that uses these criteria
