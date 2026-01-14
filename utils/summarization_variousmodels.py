import pandas as pd
from summarizer import Summarizer  # BERTSUM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer

# Load CSV
df = pd.read_csv("coursera_reviews.csv")  # Ensure columns: 'Course Name', 'Reviews'

def get_reviews_for_course(course_name):
    """Retrieve all reviews for a given course name."""
    course_reviews = df[df["course_id"] == course_name]["reviews"][:300].dropna()
    print(course_reviews[:20])
    if course_reviews.empty:
        return None
    return " ".join(course_reviews.tolist())  # Merge all reviews into a single text

def summarize_with_bertsum(text):
    """Extractive summarization using BERTSUM."""
    model = Summarizer()
    return model(text, ratio=0.3)  # Keep 30% of the text

def summarize_with_textrank(text):
    """Extractive summarization using TextRank."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 3)  # Extract top 3 sentences
    return " ".join(str(sentence) for sentence in summary)

def summarize_with_t5(text):
    """Abstractive summarization using T5."""
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = model.generate(input_ids, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_pegasus(text):
    """Abstractive summarization using PEGASUS."""
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = model.generate(input_ids, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def compute_rouge(reference, summary):
    """Compute ROUGE scores between reference and generated summary."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }

# User Input
course_name = input("Enter Course Name: ")
reviews = get_reviews_for_course(course_name)

if reviews:
    # Using first 3 sentences as the reference summary
    reference_summary = " ".join(reviews.split(". ")[:3])

    print("\n🔹 Summarization Results for:", course_name)
    
    bertsum_summary = summarize_with_bertsum(reviews)
    print("\n📌 BERTSUM (Extractive):\n", bertsum_summary)
    print("🔹 ROUGE Scores:", compute_rouge(reference_summary, bertsum_summary))

    textrank_summary = summarize_with_textrank(reviews)
    print("\n📌 TextRank (Extractive):\n", textrank_summary)
    print("🔹 ROUGE Scores:", compute_rouge(reference_summary, textrank_summary))

    t5_summary = summarize_with_t5(reviews)
    print("\n📌 T5 (Abstractive):\n", t5_summary)
    print("🔹 ROUGE Scores:", compute_rouge(reference_summary, t5_summary))

    pegasus_summary = summarize_with_pegasus(reviews)
    print("\n📌 PEGASUS (Abstractive):\n", pegasus_summary)
    print("🔹 ROUGE Scores:", compute_rouge(reference_summary, pegasus_summary))

else:
    print("Course not found!")
