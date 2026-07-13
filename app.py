import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥")

# ── Constants ─────────────────────────────────────────────────────────────────
DISCLAIMER = "\n\n⚠️ *This is general health information only. Always consult a qualified doctor for medical advice.*"
SIMILARITY_THRESHOLD = 0.3

# ── Load all 3 CSV files ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    faq_df = pd.read_csv("data/faq.csv")
    symptom_df = pd.read_csv("data/symptom_disease.csv")
    doctor_df = pd.read_csv("data/doctor_recommendation.csv")
    return faq_df, symptom_df, doctor_df

# ── Load Flan-T5 model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# ── TF-IDF search function ────────────────────────────────────────────────────
def get_tfidf_answer(user_input, faq_df):
    questions = faq_df["question"].tolist()
    all_texts = questions + [user_input]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    user_vector = tfidf_matrix[-1]
    faq_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score >= SIMILARITY_THRESHOLD:
        return faq_df["answer"].iloc[best_index], best_score
    else:
        return None, best_score

# ── Flan-T5 fallback ──────────────────────────────────────────────────────────
def ask_flan(user_input, model):
    prompt = (
        "You are a helpful healthcare assistant. "
        "Answer the following health question clearly and simply. "
        "Always remind the user to consult a doctor.\n\n"
        f"Question: {user_input}\nAnswer:"
    )
    result = model(prompt, max_new_tokens=200)
    return result[0]["generated_text"].strip()

# ── Greeting check ────────────────────────────────────────────────────────────
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return text.strip().lower() in greetings

# ── Main response function ────────────────────────────────────────────────────
def get_response(user_input, faq_df, model):
    # Layer 1: Greeting
    if is_greeting(user_input):
        return "Hello! 👋 I'm your healthcare assistant. Ask me about symptoms, conditions, or general health questions." + DISCLAIMER
    
    answer, score = get_tfidf_answer(user_input, faq_df)
    if answer:
        answer = answer.replace("⚠️", "").strip()
        return answer + DISCLAIMER

    # Layer 3: Flan-T5 fallback
    return ask_flan(user_input, model) + DISCLAIMER

# ── Load everything ───────────────────────────────────────────────────────────
faq_df, symptom_df, doctor_df = load_data()
model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏥 Healthcare Assistant")
st.caption("Ask me about symptoms, conditions, or general health advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your health question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt, faq_df, model)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})