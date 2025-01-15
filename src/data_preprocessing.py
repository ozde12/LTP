import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def preprocess_text(input_path, output_path):
    """
    Preprocesses text files by segmenting into paragraphs, cleaning extraneous characters,
    and generating target questions directly from the paragraphs.
    Args:
        input_path (str): Path to the raw text file.
        output_path (str): Path to save the processed CSV file.
    """
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
        cleaned_paragraph = re.sub(r'[^\w\s.,!?\'"-_]', '', paragraph)
        if cleaned_paragraph:  # Exclude empty paragraphs
            cleaned_paragraphs.append(cleaned_paragraph)
            # Generate a question based on the first sentence
            sentences = sent_tokenize(cleaned_paragraph)
            question = f"What is the main idea of: '{sentences[0]}'?" if sentences else "Summarize this paragraph."
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
