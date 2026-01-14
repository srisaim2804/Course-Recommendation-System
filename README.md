# NLP Project - Course Review Analysis and Recommendation System

<!-- More details here: [Presentation Link](https://www.canva.com/design/DAGj7Fte2_A/3wbyz9MFR0jGCwVRoOG3kg/view?utm_content=DAGj7Fte2_A&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h15eb9c44b4) -->


## Intent Mapping NLP Pipeline

| **Intent Code**     | **Use Sentiment Analysis?** | **Use Summarization or NLG?** |
|---------------------|-----------------------------|-------------------------------|
| `yes_no`            | ✅ Yes                      | ❌ No (direct Yes/No + sentiment-based answer) |
| `instructor`        | ✅ Yes                      | ✅ NLG (generate opinion-style sentence) |
| `content`           | ❌ No                       | ✅ Summarization (highlight topics) |
| `difficulty`        | ✅ Yes                      | ✅ NLG (generate difficulty opinion summary) |
| `career`            | ✅ Yes                      | ✅ NLG (generate relevance to job) |
| `comparison`        | ✅ Yes (if needed)          | ✅ NLG (contrastive generation using review pairs) |
| `general_opinion`   | ✅ Yes                      | ✅ Summarization or NLG (opinion summary) |
| `course_overview`   | ❌ No                       | ✅ Summarization (extractive/abstractive overview) |



## Some Examples of Questions
# List of user questions to classify
user_questions = [
    "Is this course good for beginners?",
    "How is the instructor's teaching style?",
    "What topics are covered in the course?",
    "Is this course too difficult for someone with no background?",
    "Will this course help me get a job?",
    "Which is better, Stanford’s ML course or Andrew Ng’s?",
    "What do students think about this course overall?",
    "Can you give me a quick overview of this course?"
]