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

