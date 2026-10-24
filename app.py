import streamlit as st
import pandas as pd
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Intelligent Email Assistant",
    page_icon="📧",
    layout="wide"
)


# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/emails.csv")
    df["Customer_Email"] = df["Customer_Email"].fillna("").astype(str)
    return df


df = load_data()


# -----------------------------
# TF-IDF Model
# -----------------------------

@st.cache_resource
def create_vectorizer(email_data):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(email_data)

    return vectorizer, matrix


vectorizer, tfidf_matrix = create_vectorizer(
    df["Customer_Email"].tolist()
)


# -----------------------------
# Find Similar Emails
# -----------------------------

def find_similar_emails(current_email, top_k=3):

    query_vector = vectorizer.transform(
        [current_email]
    )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "Email_ID": df.iloc[index]["Email_ID"],
            "Customer_Email": df.iloc[index]["Customer_Email"],
            "Intent": df.iloc[index]["Intent"],
            "Sentiment": df.iloc[index]["Sentiment"],
            "Priority": df.iloc[index]["Priority"],
            "Previous_Reply": df.iloc[index]["Previous_Reply"],
            "Similarity": scores[index]
        })

    return results


# -----------------------------
# Generate AI Reply
# -----------------------------

def generate_reply(current_email, similar_emails):

    context = ""

    for i, email in enumerate(similar_emails, 1):

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

-------------------------
"""

    prompt = f"""
You are an Intelligent Email Assistant for an e-commerce company.

Generate a professional reply to the current customer email.

CURRENT CUSTOMER EMAIL:
{current_email}

SIMILAR PREVIOUS INTERACTIONS:
{context}

Instructions:
- Understand the customer's intention.
- Consider the customer's sentiment.
- Use the previous replies as examples.
- Write a polite, professional and concise response.
- Do not invent order numbers, tracking numbers or other facts.
- Do not claim an action has already been completed unless supported by the context.
- Do not mention that you are an AI.
- Return only the email reply.
"""

    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )

    return response["response"]


# -----------------------------
# FRONTEND
# -----------------------------

st.title("📧 Intelligent Email Assistant")

st.write(
    "AI-powered email reply generation using "
    "previous customer interactions and Ollama."
)

st.divider()


# -----------------------------
# Dataset Information
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Emails",
        len(df)
    )

with col2:
    st.metric(
        "Categories",
        df["Category"].nunique()
    )

with col3:
    st.metric(
        "Previous Replies",
        df["Previous_Reply"].notna().sum()
    )


st.divider()


# -----------------------------
# Customer Email Input
# -----------------------------

st.subheader("✉️ Customer Email")

email = st.text_area(
    "Enter the customer's email:",
    height=150,
    placeholder="Example: My package has not arrived yet."
)


# -----------------------------
# Generate Button
# -----------------------------

if st.button(
    "🤖 Generate Professional Reply",
    type="primary"
):

    if email.strip() == "":
        st.warning("Please enter a customer email.")

    else:

        with st.spinner("Finding similar emails..."):

            similar_emails = find_similar_emails(email)


        # -----------------------------
        # Similar Interactions
        # -----------------------------

        st.subheader("🔎 Similar Previous Interactions")

        for i, item in enumerate(
            similar_emails,
            1
        ):

            with st.expander(
                f"Interaction {i} — "
                f"{item['Email_ID']} "
                f"(Similarity: {item['Similarity']:.2f})"
            ):

                st.write(
                    "**Customer Email:**"
                )

                st.write(
                    item["Customer_Email"]
                )

                st.write(
                    "**Intent:**",
                    item["Intent"]
                )

                st.write(
                    "**Sentiment:**",
                    item["Sentiment"]
                )

                st.write(
                    "**Priority:**",
                    item["Priority"]
                )

                st.write(
                    "**Previous Reply:**"
                )

                st.info(
                    item["Previous_Reply"]
                )


        # -----------------------------
        # Generate Reply
        # -----------------------------

        with st.spinner(
            "Generating professional reply with Llama 3.2..."
        ):

            reply = generate_reply(
                email,
                similar_emails
            )


        st.divider()

        st.subheader(
            "🤖 AI-Generated Professional Reply"
        )

        st.success(reply)


        # -----------------------------
        # Download Reply
        # -----------------------------

        st.download_button(
            label="⬇️ Download Reply",
            data=reply,
            file_name="generated_reply.txt",
            mime="text/plain"
        )