# app.py

import streamlit as st
#import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate suggestions using the fine-tuned model
def generate_growth_suggestion(Head):
    # Load the fine-tuned model and tokenizer
    fine_tuned_model = GPT2LMHeadModel.from_pretrained("gpt2")
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Set the padding token for the fine-tuned tokenizer
    fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token

    # Create a prompt using the year and product
    prompt = f"Explain briefly {Head}"

    # Tokenize the prompt with padding
    input_ids = fine_tuned_tokenizer.encode(prompt, return_tensors="pt", padding=True, max_length=100)

    # Generate text using the fine-tuned GPT-2 model
#    with torch.no_grad():
    output = fine_tuned_model.generate(input_ids, max_length=200, num_return_sequences=1,
                                pad_token_id=fine_tuned_tokenizer.eos_token_id,
                                no_repeat_ngram_size=2,
                                top_k=45,
                                top_p=0.95,
                                temperature=0.7,
                                do_sample=True)

#Decode the generated text
    generated_text = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

def main():
    st.title("Growth Suggestion Generator")

    # Get user input
    Head = st.text_input("Enter heading:")
    
    # Generate suggestion when the user clicks the button
    if st.button("Generate Suggestion"):
        growth_suggestion = generate_growth_suggestion(Head)
        st.subheader("Generated Suggestion:")
        st.write(growth_suggestion)

if __name__ == "__main__":
    main()
