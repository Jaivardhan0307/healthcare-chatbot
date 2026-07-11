import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Use a lightweight model
model_name = "EleutherAI/gpt-neo-125M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the text-generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Predefined medical responses
def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()

    # Predefined medical queries
    if "symptoms" in user_input_lower:
        return "Symptoms of flu include fever, cough, sore throat, and body aches. Consult a doctor for a proper diagnosis."
    elif "appointment" in user_input_lower:
        return "Sure! Please arrive at the clinic on time for your appointment. Let me know if you need to reschedule."
    elif "headache" in user_input_lower:
        return "Headaches can be caused by stress, dehydration, or other factors. Drink water, rest, and consult a doctor if persistent."
    elif "fever" in user_input_lower:
        return "Fever is commonly caused by infections like flu. Rest, hydrate, and consult a doctor if it lasts more than 3 days."
    elif "medicine" in user_input_lower:
        return "Consult with a healthcare provider to get the appropriate medication for your condition."

    else:
        # Generate a response using the model
        response = chatbot(user_input, max_length=150, num_return_sequences=1, temperature=0.7)
        return response[0]['generated_text']

# Streamlit UI
def main():
    st.title("Healthcare Assistant Bot 🤖🏥")
    
    # Display example questions to the user
    st.subheader("What can you ask?")
    st.write("""Feel free to ask your quries related to health issues """)
    
    # User input section
    user_input = st.text_area("Ask me anything about health & medicine!")

    if st.button("Submit"):
        if user_input.strip():
            with st.spinner("Processing... ⏳"):
                response = healthcare_chatbot(user_input)
            st.write("### Healthcare Assistant:", response)
        else:
            st.warning("Please enter some text!")

if __name__ == "__main__":
    main()
