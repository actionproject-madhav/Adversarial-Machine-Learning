#!/usr/bin/env python3
"""
Fix first_adoption_date and first_adoption_artifact columns.
Uses git log -S to find when each arxiv ID was ACTUALLY first added to each repo.
"""

import csv
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "artifact_repos"
CSV_FILE = BASE_DIR / "extraction_runs/run1/papers_coded_verified.csv"
OUTPUT_FILE = BASE_DIR / "extraction_runs/run1/papers_coded_verified_fixed.csv"

# Map artifact names to repo directories
ARTIFACT_REPOS = {
    "CleverHans": "cleverhans",
    "IBM ART": "adversarial-robustness-toolbox",
    "TextAttack": "TextAttack",
    "PyRIT": "PyRIT",
    "Foolbox": "foolbox",
    "HarmBench": "HarmBench",
    "AutoAttack": "auto-attack",
    "RobustBench": "robustbench",
    "MITRE ATLAS": "atlas-data",
}


def get_first_commit_date(repo_path, arxiv_id):
    """
    Get the FIRST commit date when arxiv_id was added to repo.
    Uses git log -S with --reverse to get the oldest commit first.
    """
    if not repo_path.exists():
        return None

    try:
        # Use -S to find commits that add/remove the string
        # --all to search all branches
        # --reverse to get oldest first
        # --format=%ad --date=short for YYYY-MM-DD format
        result = subprocess.run(
            ['git', 'log', '--all', '-S', arxiv_id, '--reverse',
             '--format=%ad', '--date=short'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0 and result.stdout.strip():
            # First line is the oldest (earliest) commit
            dates = result.stdout.strip().split('\n')
            earliest = dates[0]
            # Sanity check
            if earliest <= datetime.now().strftime('%Y-%m-%d'):
                return earliest
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT for {arxiv_id} in {repo_path.name}")
    except Exception as e:
        print(f"    ERROR for {arxiv_id} in {repo_path.name}: {e}")

    return None


def main():
    print("=" * 70)
    print("Fixing first_adoption_date and first_adoption_artifact")
    print("=" * 70)

    # Read existing CSV
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        papers = list(reader)

    print(f"Loaded {len(papers)} papers from {CSV_FILE.name}")
    print()

    changes = []

    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        old_date = paper['first_adoption_date']
        old_artifact = paper['first_adoption_artifact']
        found_in = paper['found_in_artifacts']

        print(f"[{i+1}/{len(papers)}] arxiv:{arxiv_id}")

        # Get artifacts that reference this paper
        artifacts = [a.strip() for a in found_in.split(',')]

        # Find adoption date for each artifact
        adoption_dates = {}
        for artifact in artifacts:
            repo_name = ARTIFACT_REPOS.get(artifact)
            if not repo_name:
                print(f"    WARNING: Unknown artifact '{artifact}'")
                continue

            repo_path = REPO_DIR / repo_name
            date = get_first_commit_date(repo_path, arxiv_id)

            if date:
                adoption_dates[artifact] = date
                print(f"    {artifact}: {date}")
            else:
                print(f"    {artifact}: NOT FOUND")

        if adoption_dates:
            # Find earliest
            earliest_artifact = min(adoption_dates.keys(), key=lambda k: adoption_dates[k])
            earliest_date = adoption_dates[earliest_artifact]

            # Build all_adoptions string (sorted by date)
            sorted_adoptions = sorted(adoption_dates.items(), key=lambda x: x[1])
            all_adoptions = '; '.join([f"{a}:{d}" for a, d in sorted_adoptions])

            # Calculate adoption lag (depends on adoption date, so we recalculate)
            pub_date = paper['paper_pub_date']
            if pub_date:
                try:
                    pub = datetime.strptime(pub_date, '%Y-%m-%d')
                    adopt = datetime.strptime(earliest_date, '%Y-%m-%d')
                    lag_months = round((adopt - pub).days / 30.44, 1)
                except:
                    lag_months = ''
            else:
                lag_months = ''

            # Check if there's a change
            if earliest_date != old_date or earliest_artifact != old_artifact:
                changes.append({
                    'arxiv_id': arxiv_id,
                    'old_date': old_date,
                    'new_date': earliest_date,
                    'old_artifact': old_artifact,
                    'new_artifact': earliest_artifact
                })
                print(f"    CHANGE: {old_artifact}:{old_date} -> {earliest_artifact}:{earliest_date}")

            # Update ONLY adoption-related columns (leave manual coding columns untouched)
            paper['first_adoption_date'] = earliest_date
            paper['first_adoption_artifact'] = earliest_artifact
            paper['all_adoptions'] = all_adoptions
            paper['adoption_lag_months'] = str(lag_months) if lag_months != '' else ''
        else:
            print(f"    WARNING: No adoption dates found!")

        print()

    # Write output
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total papers: {len(papers)}")
    print(f"Changes made: {len(changes)}")
    print(f"Output: {OUTPUT_FILE}")

    if changes:
        print("\nCHANGES:")
        for c in changes:
            print(f"  {c['arxiv_id']}: {c['old_artifact']}:{c['old_date']} -> {c['new_artifact']}:{c['new_date']}")


if __name__ == "__main__":
    main()
