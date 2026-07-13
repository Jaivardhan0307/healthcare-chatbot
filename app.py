import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_faq():
    return pd.read_csv("data/faq.csv")

@st.cache_data
def load_symptoms():
    df = pd.read_csv("data/symptom_disease.csv")
    df["symptoms_list"] = df["symptoms"].apply(
        lambda x: [s.strip().lower() for s in str(x).split(",")]
    )
    df["disease"] = df["disease"].str.strip()  # Fix trailing spaces on load
    return df

@st.cache_data
def load_doctors():
    return pd.read_csv("data/doctor_recommendation.csv")

faq_df       = load_faq()
symptom_df   = load_symptoms()
doctor_df    = load_doctors()

# ─────────────────────────────────────────────
# LOAD FLAN-T5 MODEL (Stage 1 fallback — unchanged)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()

# ─────────────────────────────────────────────
# TF-IDF SETUP (Stage 2 — unchanged)
# ─────────────────────────────────────────────

@st.cache_resource
def build_tfidf(_faq_df):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(_faq_df["question"])
    return vectorizer, question_vectors

vectorizer, question_vectors = build_tfidf(faq_df)

# ─────────────────────────────────────────────
# LAYER 1 — GREETING CHECK (Stage 1 — unchanged)
# ─────────────────────────────────────────────

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    return text.strip().lower() in greetings

# ─────────────────────────────────────────────
# LAYER 2 (NEW) — INTENT CLASSIFICATION
# What it does:
#   Checks if the user's message contains any known symptom words.
#   If yes  → treat as symptom query.
#   If no   → treat as general health question (goes to TF-IDF / Flan-T5).
#
# Why this approach:
#   Simple keyword matching against all known symptoms from our CSV.
#   No ML model needed. Fast, transparent, easy to explain in interviews.
# ─────────────────────────────────────────────

def classify_intent(user_message):
    user_message_lower = user_message.lower()

    all_symptoms = set()
    for symptoms_list in symptom_df["symptoms_list"]:
        for symptom in symptoms_list:
            # Convert underscore symptoms to spaces: high_fever → high fever
            symptom_readable = symptom.replace("_", " ")
            all_symptoms.add(symptom_readable)

    for symptom in all_symptoms:
        if symptom in user_message_lower:
            return "symptom"

    return "general"

# ─────────────────────────────────────────────
# LAYER 3 (NEW) — SYMPTOM CHECKER
# What it does:
#   Counts how many symptoms of each disease appear in the user's message.
#   The disease with the highest overlap count is the prediction.
#   Then looks up that disease in doctor_recommendation.csv.
#
# Why this approach:
#   Called "symptom overlap scoring". Simple, interpretable, no black box.
#   Perfect for a portfolio project and easy to walk through in interviews.
# ─────────────────────────────────────────────

def symptom_checker(user_message):
    user_message_lower = user_message.lower()

    best_disease = None
    best_score = 0

    for _, row in symptom_df.iterrows():
        score = 0
        for symptom in row["symptoms_list"]:
            # Convert underscore symptoms to spaces before matching
            symptom_readable = symptom.replace("_", " ")
            if symptom_readable in user_message_lower:
                score += 1

        if score > best_score:
            best_score = score
            best_disease = row["disease"].strip()  # Strip trailing spaces from disease name

    if best_score == 0 or best_disease is None:
        return (
            "I couldn't identify a specific condition from your symptoms. "
            "Please describe your symptoms in more detail, or consult a doctor directly.\n\n"
            "⚠️ *This chatbot is not a substitute for professional medical advice.*"
        )

    # Lookup doctor — strip both sides before comparing so trailing spaces don't break it
    doctor_row = doctor_df[
        doctor_df["disease"].str.strip().str.lower() == best_disease.lower()
    ]

    if doctor_row.empty:
        doctor_info = "Please consult a general physician."
        reason_info = ""
    else:
        doctor_info = doctor_row.iloc[0]["doctor_type"]
        reason_info = doctor_row.iloc[0]["reason"]

    response = (
        f"Based on your symptoms, you may have **{best_disease}**.\n\n"
        f"👨‍⚕️ **Recommended Specialist:** {doctor_info}\n\n"
        f"📋 **Why:** {reason_info}\n\n"
        f"⚠️ *This is not a medical diagnosis. Please consult a qualified doctor for proper evaluation.*"
    )

    return response

# ─────────────────────────────────────────────
# TF-IDF SEARCH (Stage 2 — unchanged)
# ─────────────────────────────────────────────

def search_faq(user_message, threshold=0.3):
    user_vec = vectorizer.transform([user_message])
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    best_idx = similarities.argmax()

    if similarities[best_idx] >= threshold:
        return faq_df.iloc[best_idx]["answer"]
    return None

# ─────────────────────────────────────────────
# FLAN-T5 FALLBACK (Stage 1 — unchanged)
# ─────────────────────────────────────────────

def ask_flan_t5(user_message):
    prompt = f"Answer this medical question clearly and simply: {user_message}"
    result = model(prompt, max_new_tokens=200)
    return result[0]["generated_text"]

# ─────────────────────────────────────────────
# MASTER RESPONSE FUNCTION
# Full routing logic — all 4 layers in order:
#   1. Greeting check
#   2. Intent classification (NEW in Stage 3)
#       → If symptom intent → symptom checker (NEW in Stage 3)
#       → If general intent → TF-IDF search (Stage 2)
#   3. TF-IDF search
#   4. Flan-T5 fallback (Stage 1)
# ─────────────────────────────────────────────

def get_response(user_message):
    # Layer 1: Greeting
    if is_greeting(user_message):
        return "Hello! 👋 I'm your Healthcare Assistant. You can ask me a medical question or describe your symptoms and I'll try to help."

    # Layer 2: Intent classification
    intent = classify_intent(user_message)

    # Layer 3: Symptom path
    if intent == "symptom":
        return symptom_checker(user_message)

    # Layer 4: General question path — TF-IDF first
    faq_answer = search_faq(user_message)
    if faq_answer:
        return faq_answer

    # Layer 5: Flan-T5 fallback
    return ask_flan_t5(user_message)

# ─────────────────────────────────────────────
# STREAMLIT UI (Stage 1 — unchanged)
# ─────────────────────────────────────────────

st.title("🏥 Healthcare Assistant")
st.markdown("*Ask a medical question or describe your symptoms.*")
st.warning("⚠️ This chatbot is for informational purposes only and is not a substitute for professional medical advice.")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Type your question or describe your symptoms...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})