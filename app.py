import streamlit as st
from transformers import pipeline

# ── MODEL SETUP ──────────────────────────────────────────────────────────────
# Flan-T5-base is an instruction-tuned model — it actually answers questions
# instead of just autocompleting text like GPT-Neo did.
# @st.cache_resource tells Streamlit: load this once, reuse it every time.
# Without this, the model would reload on every single message — very slow.
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

chatbot = load_model()


# ── SAFETY WRAPPER ────────────────────────────────────────────────────────────
# Instead of sending the raw question to Flan-T5, we wrap it in instructions.
# This tells the model to behave like a careful healthcare assistant.
def ask_flan(user_input):
    prompt = f"""You are a safe and helpful healthcare assistant.
Answer the following health question clearly and briefly.
Always remind the user to consult a doctor for proper diagnosis.

Question: {user_input}
Answer:"""
    result = chatbot(prompt, max_length=200)
    return result[0]['generated_text']


# ── KEYWORD RULES + FALLBACK ──────────────────────────────────────────────────
# Rule-based layer runs first (fast, reliable for common queries).
# If nothing matches, Flan-T5 handles it as a fallback.
DISCLAIMER = "\n\n⚠️ *This is not medical advice. Please consult a qualified doctor.*"

def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ["hi", "hello", "hey"]):
        response = "Hello! 👋 I'm your Healthcare Assistant. How can I help you today?"
    elif "symptoms" in user_input_lower:
        response = "Symptoms of flu include fever, cough, sore throat, and body aches. Consult a doctor for a proper diagnosis."
    elif "appointment" in user_input_lower:
        response = "Sure! Please arrive at the clinic on time for your appointment. Let me know if you need to reschedule."
    elif "headache" in user_input_lower:
        response = "Headaches can be caused by stress, dehydration, or other factors. Drink water, rest, and consult a doctor if persistent."
    elif "fever" in user_input_lower:
        response = "Fever is commonly caused by infections like flu. Rest, hydrate, and consult a doctor if it lasts more than 3 days."
    elif "medicine" in user_input_lower:
        response = "Consult with a healthcare provider to get the appropriate medication for your condition."
    else:
        # Flan-T5 fallback for anything not in keyword rules
        response = ask_flan(user_input)

    return response + DISCLAIMER


# ── STREAMLIT UI ──────────────────────────────────────────────────────────────
def main():
    st.title("Healthcare Assistant Bot 🤖🏥")
    st.caption("Ask me anything about health. I'll do my best to help.")

    # st.session_state is a dictionary that stays alive during your session.
    # We use it to store the full chat history so messages don't disappear.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render all previous messages from history on every rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # st.chat_input stays pinned at the bottom like a real chat app
    user_input = st.chat_input("Ask me anything about health & medicine...")

    if user_input:
        # Show user's message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Save user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = healthcare_chatbot(user_input)
            st.markdown(response)

        # Save bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()