# concat_code.py
files = [
    "__init__.py",
    "api.py",
    "classifier.py",
    "config.py",
    "models.py",
    "storage.py",
    "validator.py"
]

folder = "text_classifier"

with open("all_code.txt", "w") as outfile:
    for fname in files:
        path = f"{folder}/{fname}"
        with open(path) as infile:
            outfile.write(f"# ===== {fname} =====\n")
            outfile.write(infile.read())
            outfile.write("\n\n")
