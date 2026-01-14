# Clean generated_questions.txt by keeping only proper questions

input_path = "generated_questions.txt"
output_path = "cleaned_generated_questions.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Filter only lines that end with a question mark
cleaned_questions = [line.strip() for line in lines if line.strip().endswith("?")]

with open(output_path, "w", encoding="utf-8") as f:
    for question in cleaned_questions:
        f.write(question + "\n")

print(f"✅ Cleaned questions saved to: {output_path}")
