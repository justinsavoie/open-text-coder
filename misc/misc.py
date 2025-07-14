# Export to check
import pandas as pd
input_file = 'output/data-cps21-FULL-translated-classified.csv'
output_file = 'output/to_check.csv'
pd.read_csv(input_file)[['cps21_imp_iss', 'category']].sample(n=100, random_state=42).to_csv(output_file, index=False)
print(f"Successfully created '{output_file}' with 100 random samples.")

#

import pandas as pd
import subprocess
import re

# Load the data
input_file = 'output/data-cps21-FULL-translated-classified.csv'
df = pd.read_csv(input_file)
df = df[['cps21_imp_iss', 'category']].dropna().sample(n=20, random_state=42)

# Store LLM output
llm_outputs = []

# Loop through each row and call Ollama
for i, row in df.iterrows():
    text = row['cps21_imp_iss']
    category = row['category']
    
    prompt = f'Here is the open text: "{text}". The categorization is: "{category}". Tell me if it makes sense. Be very brief. Include a quality score out of 100 (formatted like: x / 100) at the complete end of your output.'
    
    print(f"\n[Row {i}]")
    print(prompt)

    try:
        result = subprocess.run(
            ['ollama', 'run', 'mistral:latest'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120
        )
        response = result.stdout.strip()
    except Exception as e:
        response = f"ERROR: {str(e)}"
    
    print("Response:")
    print(response)
    llm_outputs.append(response)

# Add to DataFrame
df['llm_output'] = llm_outputs

# Extract score using regex
df['score'] = df['llm_output'].str.extract(r'(\d{1,3})\s*/\s*100')[0]

# Save to file or continue working with df
df.to_csv('output/llm_checked_output.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Load data
input_file = 'output/data-cps21-FULL-translated-classified.csv'
df = pd.read_csv(input_file)

# Calculate frequencies and percentages
freq = df['category'].value_counts()
percent = freq / freq.sum() * 100

# Plot
plt.figure(figsize=(10, 6))
ax = freq.sort_values().plot(kind='barh')

# Add percentage labels
for i, (count, pct) in enumerate(zip(freq.sort_values(), percent.sort_values())):
    ax.text(count + 1, i, f'{pct:.1f}%', va='center')

# Axis and title
plt.xlabel('Frequency')
plt.ylabel('Category')
plt.title('Frequency of Categories')
plt.tight_layout()

plt.show()

# Save to PNG
plt.savefig('output/category_frequencies.png')
plt.close()

import os

def process_files(file_paths, output_path):
    # Open the output file in write mode
    with open(output_path, 'w') as outfile:
        for file_path in file_paths:
            # Extract the filename from the path
            title = os.path.basename(file_path)
            
            # Read the content of the file
            with open(file_path, 'r') as infile:
                content = infile.read()
                
            # Write the title and content to the output file
            outfile.write(f"{title}\n")
            outfile.write(content + "\n\n")

# Example usage
file_paths = [
    'README.md',
    'classifier.py',
    'latent_space_clustering.py',
    "llm_backends.py",
    'translator.py'
]
output_path = 'output/all_code.txt'

process_files(file_paths, output_path)

# concat_code.py
files = [
    "README.md",
    "llm_backends.py",
    "classifier.py",
    "translator.py"]

folder = "text_classifier"

with open("all_code.txt", "w") as outfile:
    for fname in files:
        path = f"{folder}/{fname}"
        path = fname
        with open(path) as infile:
            outfile.write(f"# ===== {fname} =====\n")
            outfile.write(infile.read())
            outfile.write("\n\n")
