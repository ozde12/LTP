import os
import re
import pandas as pd
import nltk


def preprocess_text(input_path, output_path):
    """
    Preprocesses text files by segmenting into paragraphs and cleaning extraneous characters.
    Args:
        input_path (str): Path to the raw text file.
        output_path (str): Path to save the processed CSV file.
    """
    with open(input_path, 'r', encoding="utf-8") as file:
        text = file.read()

    # Split text into paragraphs based on double newlines
    paragraphs = text.split('\n\n')

    # Clean paragraphs to remove unwanted patterns
    cleaned_paragraphs = [
        re.sub(r'[^\w\s.,!?\'"-]', '', paragraph.strip())  # Remove extraneous characters
        for paragraph in paragraphs if paragraph.strip()  # Exclude empty paragraphs
    ]

    # Save as a DataFrame
    df = pd.DataFrame({'paragraphs': cleaned_paragraphs})
    df.to_csv(output_path, index=False)
    print(f"Processed text saved to {output_path}")

if __name__ == "__main__":
    # File paths for Alice in Wonderland
    input_file = "data/raw/alice.txt"
    output_file = "data/processed/alice_paragraphs.csv"

    # Preprocess text
    preprocess_text(input_file, output_file)
