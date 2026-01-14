import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle
from tqdm import tqdm
# Load CSV
tqdm.pandas()
with open("data.pkl", "rb") as f:
    df = pickle.load(f)
# Initialize T5 model and tokenizer once
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def get_reviews_for_course(course_name):
    """Retrieve all reviews for a given course name."""
    course_reviews = df[df["course_id"] == course_name]["reviews"].dropna()
    print(len(course_reviews))
    if course_reviews.empty:
        return None
    return " ".join(course_reviews.tolist())

def chunk_text(text, max_tokens=450):
    """Split the text into manageable chunks for the model."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

def summarize_with_t5(text):
    """Abstractive summarization using T5 on full text."""
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = model.generate(input_ids, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_in_chunks(text):
    """Summarize large texts by dividing and merging summaries."""
    chunks = list(chunk_text(text))
    partial_summaries = []
    print(len(chunks))
    for chunk in tqdm(chunks):
        summary = summarize_with_t5(chunk)
        partial_summaries.append(summary)
    
    # Merge partial summaries and summarize again if needed
    combined_summary = " ".join(partial_summaries)
    
    # Optionally do a final summarization
    if len(combined_summary.split()) > 400:
        return summarize_with_t5(combined_summary)
    else:
        return combined_summary

# User Input
course_name = input("Enter Course Name: ")
reviews = get_reviews_for_course(course_name)

if reviews:
    final_summary = summarize_in_chunks(reviews)
    print("\n📌 Final Summary:\n", final_summary)
else:
    print("Course not found!")
