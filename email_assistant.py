import pandas as pd
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

FILE_PATH = "data/emails.csv"

df = pd.read_csv(FILE_PATH)

required_columns = [
    "Email_ID",
    "Customer_Name",
    "Customer_Email",
    "Intent",
    "Sentiment",
    "Previous_Reply",
    "Priority",
    "Status"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

print("Dataset loaded successfully!")
print("Number of emails:", len(df))


# --------------------------------------------------
# 2. PREPARE EMAIL TEXT
# --------------------------------------------------

df["Customer_Email"] = df["Customer_Email"].fillna("").astype(str)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    df["Customer_Email"]
)


# --------------------------------------------------
# 3. FIND SIMILAR PREVIOUS EMAILS
# --------------------------------------------------

def find_similar_emails(current_email, top_k=3):

    query_vector = vectorizer.transform([current_email])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "Email_ID": df.iloc[index]["Email_ID"],
            "Customer_Email": df.iloc[index]["Customer_Email"],
            "Intent": df.iloc[index]["Intent"],
            "Sentiment": df.iloc[index]["Sentiment"],
            "Priority": df.iloc[index]["Priority"],
            "Previous_Reply": df.iloc[index]["Previous_Reply"],
            "Similarity": similarity_scores[index]
        })

    return results


# --------------------------------------------------
# 4. GENERATE AI REPLY
# --------------------------------------------------

def generate_reply(current_email):

    similar_emails = find_similar_emails(current_email)

    context = ""

    for i, email in enumerate(similar_emails, start=1):

        context += f"""
Previous Interaction {i}

Customer Email:
{email["Customer_Email"]}

Intent:
{email["Intent"]}

Sentiment:
{email["Sentiment"]}

Priority:
{email["Priority"]}

Previous Reply:
{email["Previous_Reply"]}

Similarity Score:
{email["Similarity"]:.2f}

--------------------------------
"""

    prompt = f"""
You are an Intelligent Email Assistant for an e-commerce company.

Your task is to generate a professional reply to the customer's email.

CURRENT CUSTOMER EMAIL:
{current_email}

SIMILAR PREVIOUS CUSTOMER INTERACTIONS:
{context}

Instructions:

1. Understand the customer's intention.
2. Understand the tone and sentiment of the customer.
3. Use the previous interactions as examples.
4. Generate a professional, polite and helpful response.
5. Keep the response concise.
6. Do not invent order numbers, tracking numbers, dates or other information.
7. Do not claim that an action has already been completed unless the information supports it.
8. Do not mention that you are an AI.
9. Return only the email reply.

Generate the professional reply now.
"""

    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )

    return response["response"], similar_emails


# --------------------------------------------------
# 5. DISPLAY SIMILAR EMAILS
# --------------------------------------------------

def display_similar_emails(similar_emails):

    print("\n" + "=" * 60)
    print("SIMILAR PREVIOUS INTERACTIONS")
    print("=" * 60)

    for i, email in enumerate(similar_emails, start=1):

        print(f"\n{i}. Email ID: {email['Email_ID']}")
        print(f"Customer Email: {email['Customer_Email']}")
        print(f"Intent: {email['Intent']}")
        print(f"Sentiment: {email['Sentiment']}")
        print(f"Priority: {email['Priority']}")
        print(f"Similarity Score: {email['Similarity']:.2f}")

        print("Previous Reply:")
        print(email["Previous_Reply"])


# --------------------------------------------------
# 6. MAIN PROGRAM
# --------------------------------------------------

print("\n" + "=" * 60)
print("      INTELLIGENT EMAIL ASSISTANT")
print("=" * 60)

while True:

    current_email = input(
        "\nEnter customer email (or type 'exit'): "
    )

    if current_email.lower() == "exit":
        print("\nThank you!")
        break

    if current_email.strip() == "":
        print("Please enter an email.")
        continue

    try:

        reply, similar_emails = generate_reply(current_email)

        display_similar_emails(similar_emails)

        print("\n" + "=" * 60)
        print("AI-GENERATED PROFESSIONAL REPLY")
        print("=" * 60)

        print("\n" + reply)

    except Exception as e:

        print("\nError:", e)