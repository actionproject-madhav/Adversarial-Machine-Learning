"""
Utility module for loading and working with the benchmark criteria.
This provides a standardized way to reference the benchmark criteria anywhere in the codebase.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Path to the benchmark criteria CSV
_BENCHMARK_CSV_PATH = Path(__file__).parent / "benchmark_criteria.csv"

# Cache for loaded criteria
_criteria_cache: Optional[pd.DataFrame] = None


def load_criteria() -> pd.DataFrame:
    """Load the benchmark criteria from CSV file."""
    global _criteria_cache
    if _criteria_cache is None:
        _criteria_cache = pd.read_csv(_BENCHMARK_CSV_PATH)
    return _criteria_cache.copy()


def get_question(question_id: str) -> Dict:
    """Get a specific question by ID (e.g., 'G1', 'T1', 'Q2')."""
    criteria = load_criteria()
    question = criteria[criteria['Question_ID'] == question_id]
    if question.empty:
        raise ValueError(f"Question ID '{question_id}' not found")
    return question.iloc[0].to_dict()


def get_questions_by_category(category: str) -> pd.DataFrame:
    """Get all questions in a specific category (General, Threat Model, Query/Computation)."""
    criteria = load_criteria()
    return criteria[criteria['Category'] == category].copy()


def get_all_question_ids() -> List[str]:
    """Get a list of all question IDs in order."""
    criteria = load_criteria()
    return criteria['Question_ID'].tolist()


def get_valid_options(question_id: str) -> List[str]:
    """Get the list of valid options for a question."""
    question = get_question(question_id)
    options_str = question['Options']
    # Parse comma-separated options
    return [opt.strip() for opt in options_str.split(',')]


def build_prompt_section() -> str:
    """
    Build the prompt section for the benchmark questions.
    Returns the formatted string that can be inserted into analysis prompts.
    """
    criteria = load_criteria()
    
    prompt_parts = []
    prompt_parts.append("ANSWER THESE 12 QUESTIONS WITH ONLY THE SPECIFIED OPTIONS:\n")
    
    for _, row in criteria.iterrows():
        prompt_parts.append(
            f"{row['Question_ID']}. {row['Question_Title']} - {row['Question_Description']}\n"
            f"OPTIONS: {row['Options']}\n"
            f"ANSWER: [{row['Answer_Guidelines']}]"
        )
    
    return "\n\n".join(prompt_parts)


def build_format_section() -> str:
    """Build the format section showing expected output format."""
    criteria = load_criteria()
    
    format_parts = ["REQUIRED FORMAT:"]
    for _, row in criteria.iterrows():
        format_parts.append(row['Format_Example'])
    
    return "\n".join(format_parts)


def validate_answer(question_id: str, answer: str) -> bool:
    """Validate if an answer is valid for a given question."""
    valid_options = get_valid_options(question_id)
    return answer.strip() in valid_options


def get_criteria_summary() -> str:
    """Get a human-readable summary of all criteria."""
    criteria = load_criteria()
    
    summary = []
    summary.append("BENCHMARK CRITERIA SUMMARY")
    summary.append("=" * 60)
    
    current_category = None
    for _, row in criteria.iterrows():
        if row['Category'] != current_category:
            current_category = row['Category']
            summary.append(f"\n{current_category}:")
        
        summary.append(
            f"  {row['Question_ID']}: {row['Question_Title']} - {row['Options']}"
        )
    
    return "\n".join(summary)


# Example usage
if __name__ == "__main__":
    print(get_criteria_summary())
    print("\n" + "=" * 60)
    print("\nExample: Get question G1")
    print(get_question('G1'))
    print("\nExample: Get all General questions")
    print(get_questions_by_category('General')[['Question_ID', 'Question_Title', 'Options']])
    print("\nExample: Valid options for G2")
    print(get_valid_options('G2'))
