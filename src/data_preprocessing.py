import os
import re
import pandas as pd
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration

"""
Make sure _ * is removed from the raw txt file
Seperate the raw txt file to 80% train and 20% test
treat dialogs differently from normal paragraph
"""

nltk.download('punkt')

def generate_question(paragraph, model, tokenizer):
    """
    Generates a question for a given paragraph using a pre-trained model.
    Args:
        paragraph (str): The input paragraph.
        model: The pre-trained model for question generation.
        tokenizer: The tokenizer for the model.
    Returns:
        str: Generated question.
    """
    input_text = f"Generate a question: {paragraph}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=64, num_beams=5, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def preprocess_text(input_path, output_path, model_name="google/flan-t5-large"):
    """
    Preprocesses text files by segmenting into paragraphs, cleaning extraneous characters,
    and generating target questions using a pre-trained model.
    Args:
        input_path (str): Path to the raw text file.
        output_path (str): Path to save the processed CSV file.
        model_name (str): Name of the pre-trained model to use for question generation.
    """
    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    with open(input_path, 'r', encoding="utf-8") as file:
        text = file.read()

    # Split text into paragraphs based on double newlines
    paragraphs = text.split('\n\n')

    # Clean and filter paragraphs
    cleaned_paragraphs = []
    target_questions = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        # Skip paragraphs that start with "CHAPTER" or similar
        if re.match(r'^CHAPTER\b', paragraph, re.IGNORECASE):
            continue
        # Remove extraneous characters
        cleaned_paragraph = re.sub(r'[^\w\s.,!?\'"-]', '', paragraph)
        if cleaned_paragraph:  # Exclude empty paragraphs
            cleaned_paragraphs.append(cleaned_paragraph)
            # Generate question using the pre-trained model
            question = generate_question(cleaned_paragraph, model, tokenizer)
            target_questions.append(question)

    # Create DataFrame
    df = pd.DataFrame({
        'source_text': cleaned_paragraphs,
        'target_text': target_questions
    })

    df.to_csv(output_path, index=False)
    print(f"Processed text saved to {output_path}")

if __name__ == "__main__":
    # File paths for Alice in Wonderland
    input_file = "data/raw/alice.txt"
    output_file = "data/processed/alicesegments.csv"

    # Preprocess text
    preprocess_text(input_file, output_file)
