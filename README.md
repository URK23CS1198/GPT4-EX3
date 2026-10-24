# 📧 Intelligent Email Assistant using Ollama

## Experiment 3 - Building Applications using GPT-4
## URK23CS1198

An AI-powered Email Assistant that analyzes customer emails, retrieves similar previous interactions using TF-IDF and cosine similarity, classifies intent and sentiment, and generates professional email replies using Llama 3.2 via Ollama.

---

## 👤 Author
- **Name:** Madhavan S
- **Register Number:** URK23CS1198
- **Course:** Building Applications using GPT-4
- **Experiment:** 3 - Intelligent Email Assistant using Ollama
- **Repository:** https://github.com/URK23CS1198/GPT4-EX3

---

## 🎯 Objective

To develop an Intelligent Email Assistant using Ollama that analyzes customer emails, finds similar previous interactions, understands customer intent and sentiment, and generates professional email replies using an LLM.

---

## ❓ Problem Statement

An Email Assistant is an AI application that helps users read emails, understand customer queries, and automatically generate professional replies.

Instead of writing responses manually, the AI analyzes the customer email and suggests an appropriate response.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **TF-IDF Vectorization**
- **Cosine Similarity**
- **Ollama**
- **Llama 3.2**
- **Streamlit**

---

## 📊 Dataset

The project uses a dataset containing 100 customer email interactions (`data/emails.csv`).

Important fields include:
- `Email_ID`
- `Customer_Name`
- `Email_Date`
- `Category`
- `Priority`
- `Subject`
- `Customer_Email`
- `Intent`
- `Sentiment`
- `Previous_Reply`
- `Status`
- `Agent_Name`

---

## ✨ Key Features

1. **Dataset Integration:** Load and parse customer email dataset.
2. **Email Input:** Accept new incoming customer email text.
3. **TF-IDF Vectorization:** Convert emails into numerical vector representations.
4. **Similarity Search:** Calculate cosine similarity between the new email and previous emails.
5. **Context Retrieval:** Retrieve the most similar past interactions and previous replies.
6. **Classification:** Identify intent, sentiment, and priority.
7. **AI Reply Generation:** Generate context-aware professional replies using Llama 3.2 via Ollama.
8. **Interactive UI:** Clean Streamlit web dashboard.
9. **Export Capability:** Download the generated reply as a text file.

---

## 🔄 System Workflow

```text
Customer Email
        |
        v
Load Email Dataset
        |
        v
TF-IDF Vectorization
        |
        v
Cosine Similarity
        |
        v
Find Similar Previous Emails
        |
        v
Retrieve Intent, Sentiment and Previous Reply
        |
        v
Create Contextual Prompt
        |
        v
Ollama (Llama 3.2)
        |
        v
Generate Professional Reply
        |
        v
Display Response & Download via Streamlit
```

---

## 📁 Project Structure

```text
GPT4-EX3/
├── data/
│   └── emails.csv
├── screenshots/
│   ├── Screenshot 2026-09-02 191149.png
│   └── Screenshot 2026-09-02 191159.png
├── app.py
├── email_assistant.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📸 Screenshots

### 1. Email Assistant Interface & Analysis
![Assistant Interface](./screenshots/Screenshot%202026-09-02%20191149.png)

### 2. Context Retrieval & AI Response Generation
![AI Response](./screenshots/Screenshot%202026-09-02%20191159.png)

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/URK23CS1198/GPT4-EX3.git
cd GPT4-EX3
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Ensure Ollama is running with Llama 3.2
```bash
ollama run llama3.2
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```