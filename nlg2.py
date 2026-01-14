import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# Load reviews dataset
tqdm.pandas()
df = pd.read_csv("merged_clean_english_only.csv")  # Ensure columns: 'course_id', 'reviews'

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def get_reviews_for_course(course_name):
    """Fetch all reviews for a given course name."""
    course_reviews = df[df["course_id"] == course_name]["reviews"].dropna()
    print("Number of reviews:", len(course_reviews))
    if course_reviews.empty:
        return None
    return " ".join(course_reviews.tolist())

def chunk_text(text, max_tokens=450):
    """Chunk long review text into manageable pieces."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

def generate_review_answer_with_t5(question, context):
    """Use T5 to generate a review-style answer based on a user question and course reviews."""
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    output_ids = model.generate(
        input_ids,
        max_length=100,
        min_length=30,
        length_penalty=1.2,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_answer_from_chunks(question, text):
    """Break reviews into chunks and answer the question based on each; merge for final answer."""
    chunks = list(chunk_text(text))
    partial_answers = []
    print("Chunks:", len(chunks))
    for chunk in tqdm(chunks):
        answer = generate_review_answer_with_t5(question, chunk)
        partial_answers.append(answer)

    combined = " ".join(partial_answers)
    if len(combined.split()) > 400:
        return generate_review_answer_with_t5(question, combined)
    else:
        return combined

# --- Main Program ---
course_name = input("Enter Course Name: ")
user_question = input("Ask a question about the course: ")

reviews = get_reviews_for_course(course_name)

if reviews:
    context = reviews[:3000]  # Truncate for efficiency
    final_answer = generate_answer_from_chunks(user_question, context)
    print("\n💬 Review-Based Answer:\n", final_answer)
else:
    print("Course not found!")
