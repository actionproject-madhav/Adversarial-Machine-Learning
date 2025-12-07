"""
Split CSV into 4 Parts by Year (2022, 2023, 2024, 2025)
"""

import pandas as pd
from pathlib import Path

# Get script directory and set up paths
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "analysis_results_clean.csv"

# Load CSV
df = pd.read_csv(str(CSV_FILE))

# Split by year
part_2022 = df[df['Year'] == 2022]
part_2023 = df[df['Year'] == 2023]
part_2024 = df[df['Year'] == 2024]
part_2025 = df[df['Year'] == 2025]

# Save
part_2022.to_csv(SCRIPT_DIR / 'analysis_results_2022.csv', index=False)
part_2023.to_csv(SCRIPT_DIR / 'analysis_results_2023.csv', index=False)
part_2024.to_csv(SCRIPT_DIR / 'analysis_results_2024.csv', index=False)
part_2025.to_csv(SCRIPT_DIR / 'analysis_results_2025.csv', index=False)

print(f"2022: {len(part_2022)} papers")
print(f"2023: {len(part_2023)} papers")
print(f"2024: {len(part_2024)} papers")
print(f"2025: {len(part_2025)} papers")
print(f"Total: {len(df)} papers")