import os
from transformers import pipeline
from tqdm import tqdm

# Load question generation pipeline
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

# Folder containing review text files
reviews_folder = "reviews"

# Output file to store only questions
output_path = "generated_questions.txt"

# Ensure clean start
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Generated Questions:\n\n")

# Collect all file paths first for tqdm to work
all_files = []
for root, dirs, files in os.walk(reviews_folder):
    for filename in files:
        if filename.endswith(".txt"):
            all_files.append(os.path.join(root, filename))

# Process files with progress bar
for filepath in tqdm(all_files, desc="🔍 Processing review files"):
    with open(filepath, "r", encoding="utf-8") as review_file:
        lines = review_file.readlines()

    for line in tqdm(lines, desc=f"✏️ Generating questions from {os.path.basename(filepath)}", leave=False):
        line = line.strip()
        if not line:
            continue

        try:
            # Generate question(s)
            question_output = qg_pipeline(line, max_length=128, num_return_sequences=1)

            # Write only questions
            with open(output_path, "a", encoding="utf-8") as out_f:
                for q in question_output:
                    out_f.write(f"{q['generated_text']}\n")

        except Exception as e:
            print(f"❌ Error generating question from: {line[:60]}... | {e}")

print(f"\n✅ All questions saved to: {output_path}")
