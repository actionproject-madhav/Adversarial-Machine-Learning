import pandas as pd
import requests
import re
from time import sleep

CSV_FILE = "analysis_results_clean.csv"

# Read row 0 as header
df = pd.read_csv(CSV_FILE, header=0)

# filename column is literally named "Filename"
FILENAME_COL = "Filename"

DOI_PREFIX = "10.1145"

def extract_doi(filename):
    m = re.search(r"(\d+\.\d+)", str(filename))
    if m:
        return f"{DOI_PREFIX}/{m.group(1)}"
    return None

def get_citation(doi):
    url = f"https://doi.org/{doi}"
    headers = {"Accept": "text/x-bibliography; style=apa"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.text.strip()
    except:
        pass
    return None

# ---- Build citation list ----
citations = []

for fname in df[FILENAME_COL]:
    doi = extract_doi(fname)
    if doi:
        citation = get_citation(doi)
        print(doi, "→", citation)
        citations.append(citation)
        sleep(0.2)
    else:
        citations.append(None)

# ---- Final citation list ----
print("\nFinal citation list:\n")
for c in citations:
    print(c)
