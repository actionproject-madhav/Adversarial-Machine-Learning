import csv

# Read the CSV file
with open('analysis_results.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

# Process each row (skip header)
for i in range(1, len(rows)):
    author_text = rows[i][3]  # Column index 3 is the Authors column
    
    # Remove leading numbers and period (e.g., "17. " becomes "")
    if author_text and author_text[0].isdigit():
        # Find the first non-digit, non-period, non-space character
        clean_start = 0
        for j, char in enumerate(author_text):
            if char not in '0123456789. ':
                clean_start = j
                break
        rows[i][3] = author_text[clean_start:]

# Write back to file
with open('analysis_results_c.csv', 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print("Done! Numbers removed from Authors column.")