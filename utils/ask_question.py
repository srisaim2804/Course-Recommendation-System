import os
import pickle
import pandas as pd
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# 🧠 Load Sentence-BERT model
# ------------------------------
print("\n🔗 Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# 📁 Load Generated Questions
# ------------------------------
generated_qs_path = "generated_questions.txt"
if os.path.exists(generated_qs_path):
    with open(generated_qs_path, "r", encoding="utf-8") as f:
        generated_questions = [line.strip() for line in f if line.strip()]
    generated_embeddings = model.encode(generated_questions, convert_to_tensor=True)
else:
    generated_questions = []
    generated_embeddings = None
    print("⚠️ No 'generated_questions.txt' found or it's empty!")

# ------------------------------
# 🧾 Load Dataset
# ------------------------------
with open("data.pkl", "rb") as f:
    df = pickle.load(f)

df = df.drop_duplicates(subset=["reviews"]).reset_index(drop=True)

# ------------------------------
# 📚 Define Template Questions w/ Sentiments
# ------------------------------
question_templates = [
    ("What did students like about this course?", "positive"),
    ("What are the benefits of this course?", "positive"),
    ("What are the strengths of the course content?", "positive"),
    ("Was the teaching style effective?", "positive"),

    ("What are the problems with this course?", "negative"),
    ("What are the challenges with this course?", "negative"),
    ("What are the difficulties with this course?", "negative"),
    ("What did students dislike?", "negative"),
    ("Were there any weaknesses in the course?", "negative"),
    ("Is the course too difficult?", "negative"),

    ("What are the pros and cons of this course?", "both"),
    ("Can you summarize both good and bad experiences?", "both"),
    ("What are the strengths and weaknesses?", "both"),
]

template_texts = [q for q, _ in question_templates]
template_labels = [label for _, label in question_templates]
template_embeddings = model.encode(template_texts, convert_to_tensor=True)

# ------------------------------
# 🏫 Institution & Course Selection
# ------------------------------
institutions = sorted(df["institution"].dropna().unique())
print("\n🏫 Available Institutions:")
for i, inst in enumerate(institutions):
    print(f"{i + 1}. {inst}")

inst_index = int(input("\n🔸 Select an institution (number): ")) - 1
selected_inst = institutions[inst_index]

inst_df = df[df["institution"] == selected_inst]
courses = sorted(inst_df["name"].dropna().unique())

print(f"\n📚 Courses under {selected_inst}:")
for i, course in enumerate(courses):
    print(f"{i + 1}. {course}")

course_index = int(input("\n🔸 Select a course (number): ")) - 1
selected_course = courses[course_index]

# ------------------------------
# 🗂️ Filter Course Reviews
# ------------------------------
course_df = inst_df[inst_df["name"] == selected_course]
course_reviews = course_df["reviews"].dropna().tolist()

if not course_reviews:
    print("\n⚠️ No reviews available for this course.")
    exit()

# ------------------------------
# 💬 Ask Questions
# ------------------------------
print("\n💬 You can now ask questions about the course.")
print("📌 Type 'exit' to stop asking questions.\n")

while True:
    question = input("❓ Your question: ").strip()
    if question.lower() == "exit":
        print("\n👋 Exiting. Thanks for exploring the course reviews!")
        break

    # 🔍 Sentiment Prediction
    user_embedding = model.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, template_embeddings)[0]

    k = 3
    top_k_indices = torch.topk(cos_scores, k=k).indices.tolist()
    top_k_labels = [template_labels[i] for i in top_k_indices]
    top_k_templates = [template_texts[i] for i in top_k_indices]
    top_k_scores = [cos_scores[i].item() for i in top_k_indices]

    label_counts = Counter(top_k_labels)
    predicted_type = label_counts.most_common(1)[0][0]

    print(f"\n🧠 Top {k} template matches:")
    for i in range(k):
        print(f"   - '{top_k_templates[i]}' | Sentiment: {top_k_labels[i]} | Score: {top_k_scores[i]:.2f}")

    print(f"\n📌 Predicted sentiment: {predicted_type.upper()} (based on top-{k} voting)")

    # ------------------------------
    # 📈 Compare with Generated Questions
    # ------------------------------
    if generated_questions:
        print("\n🔍 Comparing with generated questions...")
        question_similarities = util.pytorch_cos_sim(user_embedding, generated_embeddings)[0]
        top_n = 5
        top_indices = torch.topk(question_similarities, k=top_n).indices.tolist()

        print(f"\n🧩 Top {top_n} matching generated questions:")
        for idx in top_indices:
            sim_score = question_similarities[idx].item()
            matched_q = generated_questions[idx]
            print(f"   - '{matched_q}' | Score: {sim_score:.2f}")
    else:
        print("\n⚠️ Skipping similarity with generated questions (no data).")

    print("\n" + "-" * 60)
