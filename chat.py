import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT model and tokenizer
model_path = 'gpt2' # HERE YOU CHANGE TO YOUR GPT OR ANOTHER MODEL
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Scoring function to rank the generated responses
def score_response(response):
    # You can implement your own scoring logic here
    # For example, you can use a language model to score the fluency of the response
    return len(response.split())  # Simple score based on word count

# Function to generate a response
def generate_response(query, num_responses=5):
    try:
        input_ids = tokenizer.encode(query, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=1000,
                num_return_sequences=num_responses,
                do_sample=True,
                top_k=0,
                temperature=0.7
            )

        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        scored_responses = [(response, score_response(response)) for response in responses]
        scored_responses.sort(key=lambda x: x[1], reverse=True)

        return scored_responses[0][0]  # Return the best-scoring response
    except Exception as e:
        return f'error: {e}'

# Page style
def page_style():
    # Hide Streamlit footer
    hide_streamlit_style = """
        <style>footer {visibility: hidden;}</style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title('CHAT TITLE')
    st.write('HELLO!')

    # User input
    user_input = st.text_input('TYPE YOUR QUESTION HERE!!')

    if user_input:
        # Generate response
        response = generate_response(user_input)
        st.text_area('AWNSER: ', value=response, height=250)

if __name__ == '__main__':
    page_style()
    main()
