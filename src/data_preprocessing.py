import os
import re
import pandas as pd
import nltk

nltk.download('punkt')

def preprocess_text(input_path, output_path):
    """
    Preprocesses text files by segmenting into paragraphs, cleaning extraneous characters, 
    and removing chapter headers (e.g., 'CHAPTER').
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
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        # Skip paragraphs that start with "CHAPTER" or similar
        if re.match(r'^CHAPTER\b', paragraph, re.IGNORECASE):
            continue
        # Remove extraneous characters and add to the cleaned list
        cleaned_paragraph = re.sub(r'[^\w\s.,!?\'"-]', '', paragraph)
        if cleaned_paragraph:  # Exclude empty paragraphs
            cleaned_paragraphs.append(cleaned_paragraph)

    # Add dummy "target text" for fine-tuning T5
    df = pd.DataFrame({
        'source_text': cleaned_paragraphs,
        'target_text': ["Summarize this paragraph."] * len(cleaned_paragraphs)  # Placeholder task
    })

    df.to_csv(output_path, index=False)
    print(f"Processed text saved to {output_path}")

if __name__ == "__main__":
    # File paths for Alice in Wonderland
    input_file = "data/raw/alice.txt"
    output_file = "data/processed/alicesegments.csv"

    # Preprocess text
    preprocess_text(input_file, output_file)
